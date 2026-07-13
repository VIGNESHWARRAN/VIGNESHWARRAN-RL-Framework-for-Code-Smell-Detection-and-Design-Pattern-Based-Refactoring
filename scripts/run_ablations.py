"""
SmellRL v3 — Minimal Core Ablation Study
Refactoring recommendations using SmellyCode++ dataset and 2-step DQN.

This script executes only the 4 core configurations required to substantiate
the main claims of the paper (no hyperparameter sweeps).

Configurations:
  1. full_rgat        : Proposed (RL 2-Step + RGAT + Halstead + Semantic)
  2. no_semantic      : Semantic Ablation (RL 2-Step + RGAT + Halstead only)
  3. metrics_only     : Graph Ablation (RL 2-Step + Flat Halstead metrics only)
  4. supervised_only  : Paradigm Ablation (Pre-trained RGAT + linear head, no RL)

All results are saved per-run in isolated directories under data/ablations/
"""

import argparse
import copy
import logging
import os
import sys
import json

import pandas as pd
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils      import setup_logger, get_device, CheckpointManager
from src.data       import run_preprocessing, SmellDataset, PATTERN_CLASSES
from src.models     import SmellDetectionAgent, NodeFeatureFusion
from src.training   import GCNPretrainer, DQNTrainer
from src.evaluation import classification_report_dict, SMELL_TO_PATTERN_DOMINANT

# ──────────────────────────── AblatedDataset ───────────────────────────────

class AblatedDataset(torch.utils.data.Dataset):
    """
    Zeroes out semantic embeddings dimensions to isolate structural-only features.
    """
    def __init__(self, original_dataset: SmellDataset, use_semantic: bool = True):
        self.original_dataset = original_dataset
        self.use_semantic     = use_semantic

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        item = self.original_dataset[idx]
        if not self.use_semantic:
            item = dict(item)
            item["x"] = item["x"].clone()
            # Zero out everything after the structural features (index 15 onwards)
            item["x"][:, 15:] = 0.0
        return item

# ──────────────────────────── MetricsOnlyDataset ───────────────────────────

class MetricsOnlyDataset(torch.utils.data.Dataset):
    """
    Zeroes out everything except the class node structural features to evaluate non-graph performance.
    """
    def __init__(self, original_dataset: SmellDataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        item = dict(self.original_dataset[idx])
        # Zero out semantic embeddings
        item["x"] = item["x"].clone()
        item["x"][:, 15:] = 0.0
        
        # Eliminate graph relational structure by disconnecting edges
        item["edge_index"] = torch.zeros((2, 0), dtype=torch.long)
        item["edge_type"]  = torch.zeros(0, dtype=torch.long)
        return item

# ──────────────────────────── Core Configurations ──────────────────────────

CORE_RUNS = [
    {
        "name": "full_rgat",
        "description": "Proposed (RL 2-Step + RGAT + Halstead + Semantic)",
        "use_semantic": True,
        "metrics_only": False,
        "is_rl": True,
        "gamma": 0.9
    },
    {
        "name": "no_semantic",
        "description": "Semantic Ablation (RL 2-Step + RGAT + Halstead only)",
        "use_semantic": False,
        "metrics_only": False,
        "is_rl": True,
        "gamma": 0.9
    },
    {
        "name": "metrics_only",
        "description": "Graph Ablation (RL 2-Step + Flat Halstead metrics only)",
        "use_semantic": False,
        "metrics_only": True,
        "is_rl": True,
        "gamma": 0.9
    },
    {
        "name": "supervised_only",
        "description": "Paradigm Ablation (RGAT + Semantic, No RL)",
        "use_semantic": True,
        "metrics_only": False,
        "is_rl": False,
        "gamma": 0.9 # ignored when not is_rl
    }
]

# ──────────────────────────── Helpers ──────────────────────────────────────

def _make_synthetic_ds(n: int, feature_dim: int = 783) -> SmellDataset:
    """Creates a tiny synthetic SmellDataset for dry-run testing."""
    graphs = []
    for i in range(n):
        x  = torch.randn(4, feature_dim)
        ei = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)
        et = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        y  = torch.tensor(i % 5, dtype=torch.long)
        hal = torch.randn(6)
        co_smell = torch.zeros(4)
        graphs.append({
            "x": x,
            "edge_index": ei,
            "edge_type": et,
            "y": y,
            "halstead": hal,
            "co_smell_mask": co_smell
        })
    ds = SmellDataset.__new__(SmellDataset)
    ds._graphs = graphs
    return ds

@torch.no_grad()
def evaluate_agent(agent, test_ds) -> dict:
    agent.eval()
    y_true, y_pred = [], []
    for item in test_ds:
        true_pattern = SMELL_TO_PATTERN_DOMINANT[int(item["y"].item())]
        y_true.append(true_pattern)
        action, _ = agent.select_action(item, epsilon=0.0)
        y_pred.append(action)
    agent.train()
    return classification_report_dict(y_true, y_pred, PATTERN_CLASSES)

# ──────────────────────────── Single-run executor ──────────────────────────

def run_single(
    run_spec:    dict,
    base_cfg:    dict,
    base_dir:    str,
    train_ds_raw: SmellDataset,
    val_ds_raw:   SmellDataset,
    test_ds_raw:  SmellDataset,
    device:      torch.device,
    log:         logging.Logger,
    is_dry_run:  bool = False,
) -> dict:
    log.info(f"═ Running Core Spec: {run_spec['name']} ═")
    
    # Isolation copy of config
    cfg = copy.deepcopy(base_cfg)
    run_dir = os.path.join(base_dir, run_spec["name"])
    os.makedirs(run_dir, exist_ok=True)
    cfg["paths"]["checkpoint_dir"] = run_dir

    # Adapt dataset based on run specifications
    use_semantic = run_spec.get("use_semantic", True)
    metrics_only = run_spec.get("metrics_only", False)
    
    if metrics_only:
        train_ds = MetricsOnlyDataset(train_ds_raw)
        val_ds   = MetricsOnlyDataset(val_ds_raw)
        test_ds  = MetricsOnlyDataset(test_ds_raw)
    else:
        train_ds = AblatedDataset(train_ds_raw, use_semantic=use_semantic)
        val_ds   = AblatedDataset(val_ds_raw, use_semantic=use_semantic)
        test_ds  = AblatedDataset(test_ds_raw, use_semantic=use_semantic)

    # Dry-run modifications
    if is_dry_run:
        cfg["gcn_pretrain"]["epochs"] = 2
        cfg["dqn"]["max_episodes"] = 2
        cfg["dqn"]["warmup_steps"] = 10
        cfg["dqn"]["epsilon_decay_steps"] = 20

    # 1. Supervised pre-training (always GCNPretrainer on smell index)
    log.info(f"[{run_spec['name']}] Pre-training encoder...")
    pretrainer = GCNPretrainer(cfg, device)
    pretrainer.train(train_ds, val_ds)

    # Load pre-trained encoder weights into the agent
    feature_dim = train_ds[0]["x"].shape[1] if len(train_ds) > 0 else 783
    agent = SmellDetectionAgent(cfg, feature_dim, device)
    
    gcn_ckpt = CheckpointManager(run_dir, "gcn_pretrain", log)
    payload = gcn_ckpt.load_latest()
    if not payload or "models" not in payload or "gcn" not in payload["models"]:
        raise RuntimeError("Supervised pre-training did not produce expected checkpoint.")
    agent.gcn.load_state_dict(payload["models"]["gcn"])

    is_rl = run_spec.get("is_rl", True)
    gamma = run_spec.get("gamma", 0.9)

    # 2. DQN training phase (if is_rl is True)
    if is_rl:
        log.info(f"[{run_spec['name']}] Running 2-step DQN training (gamma={gamma})...")
        cfg["dqn"]["gamma"] = gamma
        trainer = DQNTrainer(agent, cfg, device)
        trainer.train(train_ds, val_ds)
    else:
        log.info(f"[{run_spec['name']}] Skipping RL stage (supervised pre-training weights only).")
        classifier_state = payload["models"]["classifier"]
        with torch.no_grad():
            agent.q.net[-1].weight[:5] = classifier_state["weight"]
            agent.q.net[-1].bias[:5]   = classifier_state["bias"]
            agent.q.net[-1].weight[5]  = classifier_state["weight"][4]
            agent.q.net[-1].bias[5]    = classifier_state["bias"][4]

    # Evaluate on validation dataset
    val_report = evaluate_agent(agent, val_ds)
    val_f1 = val_report.get("f1", 0.0)

    # Evaluate on held-out test dataset
    test_report = evaluate_agent(agent, test_ds)
    test_f1 = test_report.get("f1", 0.0)

    log.info(f"[{run_spec['name']}] Complete. Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

    results = {
        "name":         run_spec["name"],
        "description":  run_spec["description"],
        "val_report":   val_report,
        "test_report":  test_report,
        "val_f1":       val_f1,
        "test_f1":      test_f1
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results

# ──────────────────────────── Main runner ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SmellRL v3 Minimal Ablation Suite")
    parser.add_argument("--test", action="store_true", help="Dry run mode using tiny synthetic dataset.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    log = setup_logger(cfg["paths"]["log_dir"], "SmellRL.ablations")

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║     SmellRL Minimal Core Ablation Suite (v3)    ║")
    log.info("╚══════════════════════════════════════════════════╝")

    device = get_device(cfg.get("device", "auto"))
    log.info(f"Device: {device}")

    if args.test:
        log.info("=== Smoke Test active: generating synthetic datasets ===")
        train_ds = _make_synthetic_ds(16)
        val_ds   = _make_synthetic_ds(8)
        test_ds  = _make_synthetic_ds(8)
    else:
        train_ds, val_ds, test_ds = run_preprocessing(cfg)

    base_dir = os.path.join(cfg["paths"]["results_dir"], "ablation")
    os.makedirs(base_dir, exist_ok=True)

    results_summary = []
    
    # Run the 4 core experiments in sequence
    for run_spec in CORE_RUNS:
        res = run_single(run_spec, cfg, base_dir, train_ds, val_ds, test_ds, device, log, is_dry_run=args.test)
        results_summary.append(res)

    # Compile and aggregate final reports
    summary_path = os.path.join(base_dir, "summary.csv")
    rows = []
    for r in results_summary:
        rows.append({
            "Run Name": r["name"],
            "Description": r["description"],
            "Val Macro F1": round(r["val_f1"], 4),
            "Test Macro F1": round(r["test_f1"], 4),
            "Test Accuracy": round(r["test_report"]["accuracy"], 4),
            "Test Precision": round(r["test_report"]["precision"], 4),
            "Test Recall": round(r["test_report"]["recall"], 4)
        })
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(summary_path, index=False)
    log.info(f"Aggregated ablation results summary saved to {summary_path}")

if __name__ == "__main__":
    main()
