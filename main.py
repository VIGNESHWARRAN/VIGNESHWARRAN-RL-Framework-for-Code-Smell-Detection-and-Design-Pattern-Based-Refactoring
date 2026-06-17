"""
SmellRL — main entry point.

Usage:
  python main.py                         # full pipeline (auto-resumes each stage)
  python main.py --stage preprocess      # data preprocessing only
  python main.py --stage pretrain        # GCN pre-training only
  python main.py --stage train           # RL training only
  python main.py --stage train_flat      # Flat DQN baseline training only
  python main.py --stage experiment      # run all 5 experiments only
  python main.py --stage all             # equivalent to default

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
from src.models     import HierarchicalDQNAgent, FlatDQNAgent, GCNEncoder
from src.training   import GCNPretrainer, HierarchicalTrainer, FlatDQNTrainer
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


def stage_train(cfg: dict, gcn: GCNEncoder, train_ds, val_ds,
                device: torch.device, log: logging.Logger):
    log.info("══════════════ STAGE: train SmellRL ══════════════")
    agent   = HierarchicalDQNAgent(cfg, device)

    # Load pre-trained GCN weights if available
    gcn_ckpt = CheckpointManager(cfg["paths"]["checkpoint_dir"], "gcn_pretrain", log)
    payload  = gcn_ckpt.load_latest()
    if payload and "models" in payload and "gcn" in payload["models"]:
        agent.gcn.load_state_dict(payload["models"]["gcn"])
        log.info("Loaded pre-trained GCN weights into agent")
    else:
        log.warning("No pre-trained GCN checkpoint found — starting GCN from scratch")

    trainer = HierarchicalTrainer(agent, cfg, device)
    history = trainer.train(train_ds, val_ds)
    log.info("SmellRL training complete.")
    return agent, history


def stage_train_flat(cfg: dict, gcn: GCNEncoder, train_ds,
                     device: torch.device, log: logging.Logger):
    log.info("══════════════ STAGE: train Flat DQN baseline ══════════════")
    flat_agent = FlatDQNAgent(gcn, cfg, device)
    trainer    = FlatDQNTrainer(flat_agent, cfg, device)
    history    = trainer.train(train_ds)
    log.info("Flat DQN training complete.")
    return flat_agent, history


def stage_experiment(
    cfg:        dict,
    smellrl:    HierarchicalDQNAgent,
    flat_dqn:   FlatDQNAgent,
    train_ds,
    test_ds,
    device:     torch.device,
    log:        logging.Logger,
    training_histories: dict = None,
):
    log.info("══════════════ STAGE: run experiments ══════════════")
    runner = ExperimentRunner(smellrl, flat_dqn, train_ds, test_ds, cfg, device)
    results = runner.run_all(training_histories)
    log.info("All experiments complete. Results saved to data/results/")

    # Pretty-print summary
    log.info("══════════════ RESULTS SUMMARY ══════════════")
    if "exp1" in results:
        best = max(results["exp1"].items(), key=lambda x: x[1]["f1"])
        log.info(f"  Exp1 best F1: {best[0]} → {best[1]['f1']:.4f}")
    if "exp2" in results:
        best_joint = max(results["exp2"], key=lambda x: x["Joint Accuracy"])
        log.info(f"  Exp2 best joint acc: {best_joint['Method']} → {best_joint['Joint Accuracy']:.4f}")
    if "exp3" in results:
        dmi = results["exp3"].get("Maintainability Index", {}).get("delta_mean", "N/A")
        log.info(f"  Exp3 mean ΔMI: {dmi}")

    return results


# ──────────────────────────── Full pipeline ────────────────────────────────

def run_all(cfg: dict, device: torch.device, log: logging.Logger):
    """
    Runs all stages in order.  Each stage is independently resumable.
    """
    # 1. Preprocess
    train_ds, val_ds, test_ds = stage_preprocess(cfg, log)

    # 2. GCN pre-train
    gcn = stage_pretrain_gcn(cfg, train_ds, val_ds, device, log)

    # 3. Train SmellRL
    smellrl_agent, smellrl_hist = stage_train(cfg, gcn, train_ds, val_ds, device, log)

    # 4. Train Flat DQN baseline
    flat_dqn_agent, flat_hist = stage_train_flat(cfg, gcn, train_ds, device, log)

    # 5. Experiments
    histories = {
        "SmellRL":  smellrl_hist,
        "Flat DQN": flat_hist,
    }
    stage_experiment(cfg, smellrl_agent, flat_dqn_agent, train_ds, test_ds,
                     device, log, histories)


# ──────────────────────────── Load trained models for experiment-only run ──

def _load_smellrl(cfg: dict, device: torch.device, log: logging.Logger) -> HierarchicalDQNAgent:
    agent   = HierarchicalDQNAgent(cfg, device)
    ckpt_mgr = CheckpointManager(cfg["paths"]["checkpoint_dir"], "smellrl", log)
    payload  = ckpt_mgr.load_latest()
    if payload:
        models = {
            "gcn": agent.gcn, "q1": agent.q1, "q2": agent.q2,
            "q1_target": agent.q1_target, "q2_target": agent.q2_target,
        }
        ckpt_mgr.restore(payload, models, {}, device)
        log.info("SmellRL agent loaded from checkpoint.")
    else:
        log.warning("No SmellRL checkpoint found — using untrained agent for experiments")
    return agent


def _load_flat_dqn(cfg: dict, device: torch.device, log: logging.Logger):
    gcn_cfg = cfg["gcn"]
    gcn     = GCNEncoder(gcn_cfg["node_feature_dim"], gcn_cfg["hidden_dim"],
                         gcn_cfg["output_dim"], gcn_cfg["dropout"])
    agent   = FlatDQNAgent(gcn, cfg, device)
    ckpt_mgr = CheckpointManager(cfg["paths"]["checkpoint_dir"], "flat_dqn", log)
    payload  = ckpt_mgr.load_latest()
    if payload:
        ckpt_mgr.restore(payload, {"agent": agent}, {}, device)
        log.info("Flat DQN agent loaded from checkpoint.")
        return agent
    log.warning("No Flat DQN checkpoint found.")
    return None


def _load_training_history(results_dir: str, log: logging.Logger) -> dict:
    path = os.path.join(results_dir, "training_history.json")
    if os.path.exists(path):
        with open(path) as f:
            smellrl_hist = json.load(f)
        log.info(f"Loaded SmellRL training history ({len(smellrl_hist)} episodes)")
        return {"SmellRL": smellrl_hist}
    log.warning("No training history file found — experiment 4 may be empty")
    return {}


# ──────────────────────────── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SmellRL: Two-Phase Hierarchical DQN for Code Smell Detection & Refactoring"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "preprocess", "pretrain", "train", "train_flat", "experiment"],
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    # ── Config & logging ──────────────────────────────────────────────────
    cfg    = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    log    = setup_logger("SmellRL", cfg["paths"]["log_dir"])

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║          SmellRL Experiment Pipeline             ║")
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
        gcn_cfg = cfg["gcn"]
        gcn     = GCNEncoder(gcn_cfg["node_feature_dim"], gcn_cfg["hidden_dim"],
                              gcn_cfg["output_dim"], gcn_cfg["dropout"])
        stage_train(cfg, gcn, train_ds, val_ds, device, log)

    elif args.stage == "train_flat":
        train_ds, _, _ = stage_preprocess(cfg, log)
        gcn_cfg = cfg["gcn"]
        gcn     = GCNEncoder(gcn_cfg["node_feature_dim"], gcn_cfg["hidden_dim"],
                              gcn_cfg["output_dim"], gcn_cfg["dropout"])
        # Try to load pre-trained GCN
        ckpt_mgr = CheckpointManager(cfg["paths"]["checkpoint_dir"], "gcn_pretrain", log)
        payload  = ckpt_mgr.load_latest()
        if payload and "models" in payload and "gcn" in payload["models"]:
            gcn.load_state_dict(payload["models"]["gcn"])
        stage_train_flat(cfg, gcn, train_ds, device, log)

    elif args.stage == "experiment":
        train_ds, _, test_ds = stage_preprocess(cfg, log)
        smellrl  = _load_smellrl(cfg, device, log)
        flat_dqn = _load_flat_dqn(cfg, device, log)
        histories = _load_training_history(cfg["paths"]["results_dir"], log)
        stage_experiment(cfg, smellrl, flat_dqn, train_ds, test_ds, device, log, histories)

    log.info("Done.")


if __name__ == "__main__":
    main()
