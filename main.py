 """
SmellRL — main entry point.

Usage:
  python main.py                          # full pipeline (supervised, auto-resumes)
  python main.py --stage preprocess       # data preprocessing only
  python main.py --stage train            # end-to-end supervised training (PRIMARY)
  python main.py --stage experiment       # evaluation against baselines

Ablation stages (for paper reproducibility):
  python main.py --stage pretrain         # GCN pre-training ablation only
  python main.py --stage train_dqn        # contextual-bandit DQN ablation

Each stage automatically resumes from the latest checkpoint if one exists.
Logs are written to logs/ (one file per run) and also printed to stdout.
"""

import argparse
import json
import logging
import os
import sys
import time

import yaml
import torch

from src.utils      import setup_logger, get_device, CheckpointManager
from src.data       import run_preprocessing, SmellDataset
from src.models     import SmellDetectionAgent, RGATEncoder, SupervisedSmellDetector
from src.training   import GCNPretrainer, DQNTrainer, SupervisedTrainer
from src.evaluation import ExperimentRunner


# ──────────────────────────── Config ───────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────── Stage implementations ────────────────────────

def stage_preprocess(cfg: dict, log: logging.Logger):
    log.info("══════════════ STAGE: preprocess ══════════════")
    train_ds, val_ds, test_ds = run_preprocessing(cfg)
    log.info(f"Datasets ready: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds


def stage_supervised_train(
    cfg: dict,
    train_ds,
    val_ds,
    device: torch.device,
    log: logging.Logger,
) -> tuple:
    """
    PRIMARY training stage.

    Trains RGATEncoder + linear classifier end-to-end with Focal Loss via
    SupervisedTrainer.  Returns a SupervisedSmellDetector and training history.
    """
    log.info("══════════════ STAGE: supervised train (PRIMARY) ══════════════")
    trainer  = SupervisedTrainer(cfg, device)
    detector, history = trainer.train(train_ds, val_ds)
    log.info("Supervised training complete.")
    return detector, history


def stage_pretrain_gcn(cfg: dict, train_ds, val_ds, device: torch.device, log: logging.Logger):
    """Ablation: GCN/RGAT pre-training only (no DQN, no Focal end-to-end)."""
    log.info("══════════════ STAGE: pretrain encoder (ablation) ══════════════")
    pretrainer = GCNPretrainer(cfg, device)
    enc = pretrainer.train(train_ds, val_ds)
    log.info("Encoder pre-training complete.")
    return enc


def stage_train_dqn(
    cfg: dict,
    train_ds,
    val_ds,
    device: torch.device,
    log: logging.Logger,
) -> tuple:
    """
    Ablation: contextual-bandit DQN training.

    Demonstrates why γ=0 DQN caps at ~50% accuracy on MLCQ vs. supervised
    Focal Loss. Run with --stage train_dqn to reproduce the ablation results.
    """
    log.info("══════════════ STAGE: DQN train (ablation) ══════════════")
    feature_dim = train_ds[0]["x"].shape[1] if len(train_ds) > 0 else 783
    log.info(f"Detected node feature dimension: {feature_dim}")

    agent = SmellDetectionAgent(cfg, feature_dim, device)

    # Load pre-trained encoder weights if available
    supervised_ckpt = CheckpointManager(cfg["paths"]["checkpoint_dir"], "supervised", log)
    payload = supervised_ckpt.load_latest()
    if payload and "models" in payload and "encoder" in payload["models"]:
        agent.gcn.load_state_dict(payload["models"]["encoder"])
        log.info("Loaded pre-trained encoder weights from supervised checkpoint into DQN agent")
    else:
        gcn_ckpt = CheckpointManager(cfg["paths"]["checkpoint_dir"], "gcn_pretrain", log)
        payload  = gcn_ckpt.load_latest()
        if payload and "models" in payload and "gcn" in payload["models"]:
            agent.gcn.load_state_dict(payload["models"]["gcn"])
            log.info("Loaded pre-trained encoder weights from gcn_pretrain checkpoint into DQN agent")
        else:
            log.warning("No pre-trained encoder checkpoint found — starting encoder from scratch")

    trainer = DQNTrainer(agent, cfg, device)
    history = trainer.train(train_ds, val_ds)
    log.info("DQN (ablation) training complete.")
    return agent, history


def stage_experiment(
    cfg:        dict,
    agent,                   # SupervisedSmellDetector or SmellDetectionAgent
    train_ds,
    test_ds,
    device:     torch.device,
    log:        logging.Logger,
    training_histories: dict = None,
):
    log.info("══════════════ STAGE: run experiments ══════════════")
    try:
        runner  = ExperimentRunner(agent, train_ds, test_ds, cfg, device)
        results = runner.run_all(training_histories)
    except TypeError:
        runner  = ExperimentRunner(agent, None, train_ds, test_ds, cfg, device)
        results = runner.run_all(training_histories)
    log.info("All experiments complete. Results saved to data/results/")
    return results


# ──────────────────────────── Full pipeline ────────────────────────────────

def run_all(cfg: dict, device: torch.device, log: logging.Logger):
    """
    Runs the primary pipeline in order:
      1. Preprocess  (with severity filter + edge_type v2 graph schema)
      2. Train       (SupervisedTrainer — Focal Loss, RGAT encoder)
      3. Experiment  (baseline comparison + per-class F1)

    Ablation stages (pretrain, train_dqn) are accessible individually via --stage.
    """
    # 1. Preprocess
    train_ds, val_ds, test_ds = stage_preprocess(cfg, log)

    # 2. Supervised training (PRIMARY — replaces DQN as primary pipeline)
    detector, history = stage_supervised_train(cfg, train_ds, val_ds, device, log)

    # 3. Experiments
    histories = {"SmellRL-Supervised": history}
    stage_experiment(cfg, detector, train_ds, test_ds, device, log, histories)


# ──────────────────────────── Checkpoint loaders ───────────────────────────

def _load_supervised_detector(
    cfg: dict,
    train_ds: SmellDataset,
    device: torch.device,
    log: logging.Logger,
) -> SupervisedSmellDetector:
    """Load a saved SupervisedSmellDetector from the latest checkpoint."""
    import torch.nn as nn

    feature_dim  = train_ds[0]["x"].shape[1] if len(train_ds) > 0 else 783
    gcn_cfg      = cfg["gcn"]
    n_classes    = cfg["smells"]["n_classes"]

    encoder = RGATEncoder(
        in_ch=feature_dim,
        hidden=gcn_cfg["hidden_dim"],
        out_ch=gcn_cfg["output_dim"],
        num_edge_types=gcn_cfg.get("num_edge_types", 3),
        num_heads=gcn_cfg.get("num_heads", 4),
        dropout=gcn_cfg["dropout"],
    )
    classifier = nn.Linear(gcn_cfg["output_dim"], n_classes)

    ckpt_mgr = CheckpointManager(cfg["paths"]["checkpoint_dir"], "supervised", log)
    payload  = ckpt_mgr.load_latest()
    if payload and "models" in payload:
        if "encoder" in payload["models"]:
            encoder.load_state_dict(payload["models"]["encoder"])
            classifier.load_state_dict(payload["models"]["classifier"])
            log.info("SupervisedSmellDetector loaded from checkpoint.")
        else:
            log.warning("Checkpoint found but missing 'encoder' key — using untrained model")
    else:
        log.warning("No supervised checkpoint found — using untrained model for experiments")

    return SupervisedSmellDetector(encoder, classifier, n_classes, device)


def _load_dqn_agent(
    cfg: dict,
    train_ds: SmellDataset,
    device: torch.device,
    log: logging.Logger,
) -> SmellDetectionAgent:
    """Load a saved SmellDetectionAgent (DQN ablation) from the latest checkpoint."""
    feature_dim = train_ds[0]["x"].shape[1] if len(train_ds) > 0 else 783
    agent       = SmellDetectionAgent(cfg, feature_dim, device)

    ckpt_mgr = CheckpointManager(cfg["paths"]["checkpoint_dir"], "smellrl_dqn", log)
    payload  = ckpt_mgr.load_latest()
    if payload:
        if "models" in payload and "agent" in payload["models"]:
            agent.load_state_dict(payload["models"]["agent"])
        log.info("SmellDetectionAgent (DQN ablation) loaded from checkpoint.")
    else:
        log.warning("No DQN checkpoint found — using untrained agent")
    return agent


def _load_training_history(results_dir: str, log: logging.Logger) -> dict:
    path = os.path.join(results_dir, "training_history.json")
    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)
        log.info(f"Loaded training history ({len(history)} entries)")
        return {"SmellRL-Supervised": history}
    log.warning("No training history file found.")
    return {}


# ──────────────────────────── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SmellRL: RGAT + Focal Loss for Code Smell Detection"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",          # PRIMARY: preprocess → supervised train → experiment
            "preprocess",
            "train",        # PRIMARY: SupervisedTrainer (Focal Loss + RGAT)
            "experiment",
            # ── Ablation stages ──────────────────────────────────────────────
            "pretrain",     # ABLATION: encoder-only pre-training
            "train_dqn",    # ABLATION: contextual-bandit DQN
        ],
        help=(
            "Pipeline stage to run (default: all). "
            "Use 'train' for the primary supervised pipeline. "
            "Use 'train_dqn' / 'pretrain' for ablation baselines."
        ),
    )
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    log    = setup_logger("SmellRL", cfg["paths"]["log_dir"])

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║     SmellRL  —  RGAT + Focal Loss Pipeline      ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info(f"Config: {args.config}")
    log.info(f"Stage:  {args.stage}")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(device)}")

    # ── Dispatch ─────────────────────────────────────────────────────────
    if args.stage == "all":
        run_all(cfg, device, log)

    elif args.stage == "preprocess":
        stage_preprocess(cfg, log)

    elif args.stage == "train":
        train_ds, val_ds, _ = stage_preprocess(cfg, log)
        stage_supervised_train(cfg, train_ds, val_ds, device, log)

    elif args.stage == "experiment":
        train_ds, _, test_ds = stage_preprocess(cfg, log)
        detector   = _load_supervised_detector(cfg, train_ds, device, log)
        histories  = _load_training_history(cfg["paths"]["results_dir"], log)
        stage_experiment(cfg, detector, train_ds, test_ds, device, log, histories)

    # ── Ablation stages ──────────────────────────────────────────────────
    elif args.stage == "pretrain":
        train_ds, val_ds, _ = stage_preprocess(cfg, log)
        stage_pretrain_gcn(cfg, train_ds, val_ds, device, log)

    elif args.stage == "train_dqn":
        train_ds, val_ds, _ = stage_preprocess(cfg, log)
        stage_train_dqn(cfg, train_ds, val_ds, device, log)

    log.info("Done.")


if __name__ == "__main__":
    main()