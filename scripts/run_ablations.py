"""
Run ablation studies and parameter sweeps for SmellRL.

Supports:
  1. Isolated loop per run config: runs preprocessing, GCN pre-training, DQN training, and baseline experiments.
  2. GCN structural ablations (Full 2-layer, No GCN, 1-layer GCN).
  3. DQN hyperparameter sweeps (episodes, epsilon decay, learning rates, etc.).
  4. Embedding toggle detection (auto-switches directories to avoid cache/results clash).
  5. Dynamic virtual vs AST graph checking using zero-embedding detection.
  6. Saves comprehensive run configuration, training/test results, and graph statistics to data/results/.

Usage:
  python scripts/run_ablations.py [--config config.yaml] [--test]
"""

import argparse
import copy
import logging
import os
import sys
import json
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils      import setup_logger, get_device, CheckpointManager
from src.data       import run_preprocessing, SmellDataset, SMELL_CLASSES
from src.models     import SmellDetectionAgent, GCNEncoder, GCNLayer
from src.training   import DQNTrainer, GCNPretrainer
from src.evaluation import classification_report_dict, ExperimentRunner
from src.embeddings import USE_EMBEDDINGS_MODEL


# ─── Configuration Runs Definition ──────────────────────────────────────────
# Users can modify this list of dictionaries to run any combinations of parameters.
RUNS = [
    {
        "name": "full_smellrl",
        "description": "Standard SmellRL (2-layer GCN, default hyperparameters)",
        "gcn_type": "full",
        "dqn.max_episodes": 100,
        "dqn.epsilon_decay_steps": 50000,
        "dqn.learning_rate": 0.0005,
    },
    {
        "name": "no_gcn",
        "description": "Ablation: No GCN (metric-vector only state)",
        "gcn_type": "no_gcn",
        "dqn.max_episodes": 100,
        "dqn.epsilon_decay_steps": 50000,
        "dqn.learning_rate": 0.0005,
    },
    {
        "name": "gcn_1layer",
        "description": "Ablation: 1-layer GCN",
        "gcn_type": "gcn_1layer",
        "dqn.max_episodes": 100,
        "dqn.epsilon_decay_steps": 50000,
        "dqn.learning_rate": 0.0005,
    },
    {
        "name": "fast_epsilon_decay",
        "description": "Hyperparameter check: Faster Epsilon Decay (25k steps)",
        "gcn_type": "full",
        "dqn.max_episodes": 100,
        "dqn.epsilon_decay_steps": 25000,
        "dqn.learning_rate": 0.0005,
    },
    {
        "name": "slow_epsilon_decay",
        "description": "Hyperparameter check: Slower Epsilon Decay (100k steps)",
        "gcn_type": "full",
        "dqn.max_episodes": 100,
        "dqn.epsilon_decay_steps": 100000,
        "dqn.learning_rate": 0.0005,
    },
    {
        "name": "low_lr",
        "description": "Hyperparameter check: Lower Learning Rate (0.0001)",
        "gcn_type": "full",
        "dqn.max_episodes": 100,
        "dqn.epsilon_decay_steps": 50000,
        "dqn.learning_rate": 0.0001,
    },
]


# ─── Identity GCN Ablation Module ───────────────────────────────────────────

class IdentityGCN(GCNEncoder):
    """Replaces graph convolution with a plain MLP on the class node only (no message passing)."""
    def __init__(self, in_ch: int, hidden: int = 128, out_ch: int = 128, dropout: float = 0.1):
        super().__init__(in_ch, hidden, out_ch, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Only use the class node (index 0) — no message passing
        h = x[0:1]                      # Shape: [1, in_ch]
        h = F.relu(self.conv1.linear(h))
        h = self.conv2.linear(h)
        return h                        # Shape: [1, out_ch]


# ─── 1-Layer GCN Ablation Module ────────────────────────────────────────────

class OneLayerGCN(GCNEncoder):
    """A GCN encoder with only 1 layer of message passing."""
    def __init__(self, in_ch: int, out_ch: int = 128, dropout: float = 0.1):
        super().__init__(in_ch, out_ch, out_ch, dropout)
        # Re-define structure to use a single layer directly
        self.conv1 = GCNLayer(in_ch, out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.dropout(h)
        return h.mean(dim=0, keepdim=True)


# ─── Dynamic Config Update Helper ───────────────────────────────────────────

def update_config_path(key: str, value, config_dict: dict):
    """Helper to update nested dictionary key like 'dqn.max_episodes'."""
    parts = key.split('.')
    d = config_dict
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


# ─── Graph Type Checking & Statistics Helpers ───────────────────────────────

def is_graph_virtual(graph) -> bool:
    """Detects if a graph is virtual based on zero-valued embeddings for method/field nodes."""
    x = graph["x"]
    if x.shape[0] <= 1:
        # Only class node present. Without methods/fields we check if javalang is missing.
        try:
            import javalang
            return False
        except ImportError:
            return True
    
    # In virtual graphs, the method and field node embeddings (index 9 onwards) are all zeros.
    embedding_part = x[1:, 9:]
    return bool(torch.all(embedding_part == 0.0).item())


def collect_dataset_stats(dataset, dataset_name: str) -> dict:
    """Computes virtual/AST counts and per-class distributions."""
    total = len(dataset)
    virtual_count = 0
    class_counts = {s: 0 for s in SMELL_CLASSES}
    virtual_class_counts = {s: 0 for s in SMELL_CLASSES}
    
    for item in dataset:
        true_idx = int(item["y"].item())
        smell_name = SMELL_CLASSES[true_idx]
        class_counts[smell_name] += 1
        
        is_virt = is_graph_virtual(item)
        if is_virt:
            virtual_count += 1
            virtual_class_counts[smell_name] += 1
            
    return {
        "total_graphs": total,
        "virtual_graphs": virtual_count,
        "ast_graphs": total - virtual_count,
        "class_distribution": class_counts,
        "virtual_class_distribution": virtual_class_counts
    }


# ─── Train and Evaluate Run Variants ────────────────────────────────────────

def train_variant(
    run_cfg: dict,
    run_cfg_dict: dict,
    train_ds: SmellDataset,
    device: torch.device,
    log: logging.Logger
) -> tuple:
    """Trains a single variant with overridden parameters."""
    feature_dim = train_ds[0]["x"].shape[1] if len(train_ds) > 0 else run_cfg_dict["ck_metrics"]["n_features"]
    log.info(f"[Ablation Run] Initializing SmellDetectionAgent (feature_dim={feature_dim})")
    
    agent = SmellDetectionAgent(run_cfg_dict, feature_dim, device)

    # Apply structural GCN overrides/load pretrained GCN weights
    gcn_type = run_cfg.get("gcn_type", "full")
    gcn_cfg = run_cfg_dict["gcn"]

    if gcn_type == "no_gcn":
        log.info("[Ablation Run] Overriding GCN with IdentityGCN (no message passing)")
        agent.gcn = IdentityGCN(feature_dim, gcn_cfg["hidden_dim"], gcn_cfg["output_dim"], gcn_cfg["dropout"]).to(device)
    elif gcn_type == "gcn_1layer":
        log.info("[Ablation Run] Overriding GCN with OneLayerGCN (1-layer message passing)")
        agent.gcn = OneLayerGCN(feature_dim, gcn_cfg["output_dim"], gcn_cfg["dropout"]).to(device)
    else:
        # Load pre-trained GCN if available
        gcn_ckpt = CheckpointManager(run_cfg_dict["paths"]["checkpoint_dir"], "gcn_pretrain", log)
        payload  = gcn_ckpt.load_latest()
        if payload and "models" in payload and "gcn" in payload["models"]:
            agent.gcn.load_state_dict(payload["models"]["gcn"])
            log.info("[Ablation Run] Loaded pre-trained GCN weights into agent")
        else:
            log.warning("[Ablation Run] No pre-trained GCN checkpoint found — starting GCN from scratch")

    trainer = DQNTrainer(agent, run_cfg_dict, device)
    history = trainer.train(train_ds)
    return agent, history


@torch.no_grad()
def evaluate_agent(agent: SmellDetectionAgent, test_ds: SmellDataset) -> dict:
    """Evaluates the trained agent on the test dataset directly as a backup."""
    agent.eval()
    y_true = []
    y_pred = []
    for item in test_ds:
        true_y = int(item["y"].item())
        action, _ = agent.select_action(item, epsilon=0.0)
        y_true.append(true_y)
        y_pred.append(action)
    agent.train()
    
    report = classification_report_dict(y_true, y_pred, SMELL_CLASSES)
    return report


# ─── Main Execution Pipeline ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SmellRL Ablation Studies & Parameter Sweeps")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--test", action="store_true", help="Run a quick test of the pipeline")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    device = get_device(base_cfg.get("device", "auto"))
    log = setup_logger("SmellRL.ablations", base_cfg["paths"]["log_dir"])

    log.info("══════════════════════════════════════════════════")
    log.info("       SmellRL Ablations & Parameter Sweeps       ")
    log.info("══════════════════════════════════════════════════")
    log.info(f"Embeddings toggle (USE_EMBEDDINGS_MODEL): {USE_EMBEDDINGS_MODEL}")
    log.info(f"Device: {device}")
    if args.test:
        log.info("[TEST MODE ACTIVE] Slicing datasets and setting minimal training episodes.")

    # Base directories to isolate
    emb_suffix = "embeddings" if USE_EMBEDDINGS_MODEL else "no_embeddings"
    base_processed_dir = os.path.join(base_cfg["paths"]["processed_dir"], emb_suffix)
    base_checkpoint_dir = os.path.join(base_cfg["paths"]["checkpoint_dir"], emb_suffix)
    base_results_dir = os.path.join(base_cfg["paths"]["results_dir"], emb_suffix)

    # Adjust configurations if test mode is enabled
    runs_to_execute = RUNS
    if args.test:
        runs_to_execute = [
            {
                "name": "test_full_gcn",
                "description": "Test Run with Full GCN",
                "gcn_type": "full",
                "dqn.max_episodes": 2,
                "dqn.epsilon_decay_steps": 10,
                "dqn.learning_rate": 0.0005,
                "dqn.warmup_steps": 2,
                "dqn.batch_size": 2,
            },
            {
                "name": "test_no_gcn",
                "description": "Test Run with No GCN",
                "gcn_type": "no_gcn",
                "dqn.max_episodes": 2,
                "dqn.epsilon_decay_steps": 10,
                "dqn.learning_rate": 0.0005,
                "dqn.warmup_steps": 2,
                "dqn.batch_size": 2,
            },
            {
                "name": "test_gcn_1layer",
                "description": "Test Run with 1-layer GCN",
                "gcn_type": "gcn_1layer",
                "dqn.max_episodes": 2,
                "dqn.epsilon_decay_steps": 10,
                "dqn.learning_rate": 0.0005,
                "dqn.warmup_steps": 2,
                "dqn.batch_size": 2,
            }
        ]

    results = []

    # Run parameter sweeps & ablations
    for idx, run_cfg in enumerate(runs_to_execute):
        run_name = run_cfg["name"]
        description = run_cfg["description"]
        log.info(f"\n[Run {idx+1}/{len(runs_to_execute)}] Starting: {run_name} ({description})")
        
        start_time = time.time()

        # Set up isolated directories specifically for this configuration run
        run_cfg_dict = copy.deepcopy(base_cfg)
        run_cfg_dict["paths"]["processed_dir"] = os.path.join(base_processed_dir, run_name)
        run_cfg_dict["paths"]["checkpoint_dir"] = os.path.join(base_checkpoint_dir, run_name)
        run_cfg_dict["paths"]["results_dir"] = os.path.join(base_results_dir, run_name)

        if USE_EMBEDDINGS_MODEL:
            run_cfg_dict["semantic"]["cache_dir"] = os.path.join(run_cfg_dict["semantic"]["cache_dir"], "real")
        else:
            run_cfg_dict["semantic"]["cache_dir"] = os.path.join(run_cfg_dict["semantic"]["cache_dir"], "synthetic")

        # Disable saving intermediate checkpoints to prevent overwriting/clutter
        run_cfg_dict["training"]["checkpoint_freq"] = 9999

        # Apply parameter overrides to run_cfg_dict
        for k, v in run_cfg.items():
            if k in ["name", "description", "gcn_type"]:
                continue
            update_config_path(k, v, run_cfg_dict)

        # Apply test mode overrides if test is active
        if args.test:
            run_cfg_dict["gcn_pretrain"]["epochs"] = 2
            run_cfg_dict["gcn_pretrain"]["checkpoint_freq"] = 9999

        log.info(f"  Processed data dir: {run_cfg_dict['paths']['processed_dir']}")
        log.info(f"  Checkpoints dir:    {run_cfg_dict['paths']['checkpoint_dir']}")
        log.info(f"  Results dir:        {run_cfg_dict['paths']['results_dir']}")

        # ─── Stage 1: Preprocessing ───
        train_ds, val_ds, test_ds = run_preprocessing(run_cfg_dict)

        # ─── Stage 2: Gather Dataset/Graph Statistics ───
        train_stats = collect_dataset_stats(train_ds, "train")
        val_stats = collect_dataset_stats(val_ds, "val")
        test_stats = collect_dataset_stats(test_ds, "test")

        log.info(f"  Graphs loaded: train={len(train_ds)} (virtual: {train_stats['virtual_graphs']}), test={len(test_ds)} (virtual: {test_stats['virtual_graphs']})")

        # Slice datasets if test mode is active to speed up training execution
        if args.test:
            train_ds._graphs = train_ds._graphs[:16]
            val_ds._graphs = val_ds._graphs[:8]
            test_ds._graphs = test_ds._graphs[:8]

        # ─── Stage 3: GCN Pre-training ───
        gcn_type = run_cfg.get("gcn_type", "full")
        if gcn_type != "no_gcn":
            log.info(f"[Ablation Run] Starting GCN pretraining for {run_cfg_dict['gcn_pretrain']['epochs']} epochs...")
            pretrainer = GCNPretrainer(run_cfg_dict, device)
            pretrainer.train(train_ds, val_ds)

        # ─── Stage 4: DQN Agent Training ───
        agent, history = train_variant(run_cfg, run_cfg_dict, train_ds, device, log)
        elapsed_time = time.time() - start_time

        # ─── Stage 5: Experiment Baseline Runner & Evaluation ───
        log.info("[Ablation Run] Running baseline comparison and F1 breakdown...")
        runner = ExperimentRunner(agent, train_ds, test_ds, run_cfg_dict, device)
        exp_results = runner.run_all(training_histories={"SmellRL": history})

        # Retrieve SmellRL metrics from runner summary
        report = exp_results["baselines"].get("SmellRL (Ours)", {})
        if not report:
            report = evaluate_agent(agent, test_ds)

        log.info(f"[Run {idx+1} Completed] Test Acc={report.get('accuracy', 0.0):.4f} | Macro F1={report.get('f1', 0.0):.4f} | Time={elapsed_time:.1f}s")

        # Compile comprehensive result dictionary containing all stats and configs
        final_history_step = history[-1] if len(history) > 0 else {}
        run_result = {
            "run_index": idx + 1,
            "run_name": run_name,
            "description": description,
            "embeddings_enabled": USE_EMBEDDINGS_MODEL,
            "elapsed_seconds": round(elapsed_time, 2),
            "hyperparameters": {k: v for k, v in run_cfg.items() if k not in ["name", "description"]},
            "dataset_statistics": {
                "train": train_stats,
                "val": val_stats,
                "test": test_stats
            },
            "metrics": {
                "accuracy": round(report.get("accuracy", 0.0), 4),
                "macro_f1": round(report.get("f1", 0.0), 4),
                "macro_precision": round(report.get("precision", 0.0), 4),
                "macro_recall": round(report.get("recall", 0.0), 4),
                "per_class_f1": {
                    smell: round(report.get("per_class", {}).get(smell, {}).get("f1", 0.0), 4)
                    for smell in SMELL_CLASSES
                }
            },
            "training_summary": {
                "final_loss": round(final_history_step.get("loss", 0.0), 6),
                "final_reward": round(final_history_step.get("reward", 0.0), 4),
                "final_accuracy": round(final_history_step.get("accuracy", 0.0), 4),
                "total_episodes": len(history)
            },
            "config": run_cfg_dict
        }
        results.append(run_result)

    # Save overall aggregated results
    os.makedirs(base_results_dir, exist_ok=True)
    file_prefix = "ablation_results" if not args.test else "test_ablation_results"
    
    json_path = os.path.join(base_results_dir, f"{file_prefix}_{emb_suffix}.json")
    csv_path = os.path.join(base_results_dir, f"{file_prefix}_{emb_suffix}.csv")

    # Save detailed JSON file
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Successfully saved aggregated JSON results to: {json_path}")

    # Flatten results for CSV sheet
    csv_rows = []
    for r in results:
        flat_row = {
            "run_index": r["run_index"],
            "run_name": r["run_name"],
            "embeddings_enabled": r["embeddings_enabled"],
            "elapsed_seconds": r["elapsed_seconds"],
            
            # Dataset counts
            "train_total_graphs": r["dataset_statistics"]["train"]["total_graphs"],
            "train_virtual_graphs": r["dataset_statistics"]["train"]["virtual_graphs"],
            "train_ast_graphs": r["dataset_statistics"]["train"]["ast_graphs"],
            
            "test_total_graphs": r["dataset_statistics"]["test"]["total_graphs"],
            "test_virtual_graphs": r["dataset_statistics"]["test"]["virtual_graphs"],
            "test_ast_graphs": r["dataset_statistics"]["test"]["ast_graphs"],
            
            # Key performance metrics
            "accuracy": r["metrics"]["accuracy"],
            "macro_f1": r["metrics"]["macro_f1"],
            "macro_precision": r["metrics"]["macro_precision"],
            "macro_recall": r["metrics"]["macro_recall"],
            "final_training_loss": r["training_summary"]["final_loss"],
            "final_training_reward": r["training_summary"]["final_reward"],
            "total_episodes": r["training_summary"]["total_episodes"],
        }
        # Add hyperparams to columns
        for hp_k, hp_v in r["hyperparameters"].items():
            flat_row[f"hp_{hp_k}"] = hp_v
        # Add per class F1 to columns
        for smell, f1_val in r["metrics"]["per_class_f1"].items():
            flat_row[f"f1_{smell}"] = f1_val
        csv_rows.append(flat_row)

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    log.info(f"Successfully saved aggregated CSV results to:  {csv_path}")


if __name__ == "__main__":
    main()
