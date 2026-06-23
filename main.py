"""
SmellRL — main entry point.

Usage:
  python main.py                         # full pipeline (auto-resumes each stage)
  python main.py --stage preprocess      # data preprocessing only
  python main.py --stage pretrain        # GCN pre-training only
  python main.py --stage train           # RL training only
  python main.py --stage experiment      # run evaluation and ablation study

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
from src.models     import SmellDetectionAgent, GCNEncoder
from src.training   import GCNPretrainer, DQNTrainer
# Note: Ensure your evaluation.py has an ExperimentRunner or AblationRunner configured for the 5-class setup
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


def stage_pretrain_gcn(cfg: dict, train_ds, val_ds, device: torch.device, log: logging.Logger):
    log.info("══════════════ STAGE: pretrain GCN ══════════════")
    pretrainer = GCNPretrainer(cfg, device)
    gcn = pretrainer.train(train_ds, val_ds)
    log.info("GCN pre-training complete. Encoder ready.")
    return gcn


def stage_train(cfg: dict, train_ds, val_ds, device: torch.device, log: logging.Logger):
    log.info("══════════════ STAGE: train SmellRL ══════════════")
    
    # Dynamically detect feature dimension from the built dataset
    feature_dim = train_ds[0]["x"].shape[1] if len(train_ds) > 0 else cfg["ck_metrics"]["n_features"]
    log.info(f"Detected node feature dimension: {feature_dim}")
    
    agent = SmellDetectionAgent(cfg, feature_dim, device)

    # Load pre-trained GCN weights if available
    gcn_ckpt = CheckpointManager(cfg["paths"]["checkpoint_dir"], "gcn_pretrain", log)
    payload  = gcn_ckpt.load_latest()
    if payload and "models" in payload and "gcn" in payload["models"]:
        agent.gcn.load_state_dict(payload["models"]["gcn"])
        log.info("Loaded pre-trained GCN weights into agent")
    else:
        log.warning("No pre-trained GCN checkpoint found — starting GCN from scratch")

    trainer = DQNTrainer(agent, cfg, device)
    # The new DQNTrainer handles validation natively if configured
    history = trainer.train(train_ds) 
    log.info("SmellRL training complete.")
    return agent, history


def stage_experiment(
    cfg:        dict,
    agent:      SmellDetectionAgent,
    train_ds,
    test_ds,
    device:     torch.device,
    log:        logging.Logger,
    training_histories: dict = None,
):
    log.info("══════════════ STAGE: run experiments ══════════════")
    
    # Instantiate the runner (assuming evaluation.py has been updated to remove flat_dqn)
    # Fallback to standard instantiation depending on your specific evaluation.py signature
    try:
        runner = ExperimentRunner(agent, train_ds, test_ds, cfg, device)
        results = runner.run_all(training_histories)
    except TypeError:
        # If evaluation.py still requires a second agent argument (like flat_dqn), pass None
        runner = ExperimentRunner(agent, None, train_ds, test_ds, cfg, device)
        results = runner.run_all(training_histories)
        
    log.info("All experiments complete. Results saved to data/results/")
    return results


# ──────────────────────────── Full pipeline ────────────────────────────────

def run_all(cfg: dict, device: torch.device, log: logging.Logger):
    """
    Runs all stages in order.  Each stage is independently resumable.
    """
    # 1. Preprocess
    train_ds, val_ds, test_ds = stage_preprocess(cfg, log)

    # 2. GCN pre-train
    stage_pretrain_gcn(cfg, train_ds, val_ds, device, log)

    # 3. Train SmellRL
    agent, history = stage_train(cfg, train_ds, val_ds, device, log)

    # 4. Experiments
    histories = {
        "SmellRL":  history,
    }
    stage_experiment(cfg, agent, train_ds, test_ds, device, log, histories)


# ──────────────────────────── Load trained models for experiment-only run ──

def _load_agent(cfg: dict, train_ds: SmellDataset, device: torch.device, log: logging.Logger) -> SmellDetectionAgent:
    feature_dim = train_ds[0]["x"].shape[1] if len(train_ds) > 0 else cfg["ck_metrics"]["n_features"]
    agent   = SmellDetectionAgent(cfg, feature_dim, device)
    
    ckpt_mgr = CheckpointManager(cfg["paths"]["checkpoint_dir"], "smellrl", log)
    payload  = ckpt_mgr.load_latest()
    if payload:
        # Depending on how the new DQNTrainer saves the agent
        if "models" in payload and "agent" in payload["models"]:
            agent.load_state_dict(payload["models"]["agent"])
        elif "models" in payload and "q" in payload["models"]:
            agent.q.load_state_dict(payload["models"]["q"])
            agent.gcn.load_state_dict(payload["models"]["gcn"])
        log.info("SmellRL agent loaded from checkpoint.")
    else:
        log.warning("No SmellRL checkpoint found — using untrained agent for experiments")
    return agent


def _load_training_history(results_dir: str, log: logging.Logger) -> dict:
    path = os.path.join(results_dir, "training_history.json")
    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)
        log.info(f"Loaded SmellRL training history ({len(history)} episodes)")
        return {"SmellRL": history}
    log.warning("No training history file found.")
    return {}


# ──────────────────────────── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Semantic-Aware SmellRL: GCN + DQN for Code Smell Detection"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "preprocess", "pretrain", "train", "experiment"],
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    # ── Config & logging ──────────────────────────────────────────────────
    cfg    = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    log    = setup_logger("SmellRL", cfg["paths"]["log_dir"])

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║          Semantic-Aware SmellRL Pipeline         ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info(f"Config: {args.config}")
    log.info(f"Stage:  {args.stage}")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(device)}")

    # ── Dispatch ──────────────────────────────────────────────────────────
    if args.stage == "all":
        run_all(cfg, device, log)

    elif args.stage == "preprocess":
        stage_preprocess(cfg, log)

    elif args.stage == "pretrain":
        train_ds, val_ds, _ = stage_preprocess(cfg, log)
        stage_pretrain_gcn(cfg, train_ds, val_ds, device, log)

    elif args.stage == "train":
        train_ds, val_ds, _ = stage_preprocess(cfg, log)
        stage_train(cfg, train_ds, val_ds, device, log)

    elif args.stage == "experiment":
        train_ds, _, test_ds = stage_preprocess(cfg, log)
        agent = _load_agent(cfg, train_ds, device, log)
        histories = _load_training_history(cfg["paths"]["results_dir"], log)
        stage_experiment(cfg, agent, train_ds, test_ds, device, log, histories)

    log.info("Done.")


if __name__ == "__main__":
    main()