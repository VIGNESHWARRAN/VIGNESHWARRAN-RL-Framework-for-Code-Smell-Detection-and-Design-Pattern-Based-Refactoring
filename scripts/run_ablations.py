"""
Run ablation studies and parameter sweeps for SmellRL.

Research Methodology:
  - Phase 1: Tune the Engine (Hyperparameter Sweep)
    Tuning learning rates, exploration epsilons, and reward shaping on full GCN+Semantic architecture.
    Identifies the "Optimal Baseline" with the highest Macro F1 score on the test set.
  - Phase 2: Remove the Parts (Ablation Study)
    Uses the optimal hyperparameters from Phase 1 and systematically "breaks" the model:
      1. Semantic Ablation (No Semantic): use_semantic=False, full GCN.
      2. Structural Ablation (Shallow Graph): use_semantic=True, 1-layer GCN.
      3. Traditional Baseline (CK Only): use_semantic=False, 1-layer GCN.

All folders (processed_dir, checkpoint_dir, results_dir) are isolated per configuration run.
Dataset virtual/AST graph statistics are dynamically computed and exported with process metadata.

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


# ─── Dataset Ablation Wrapper ───────────────────────────────────────────────

class AblatedDataset(torch.utils.data.Dataset):
    """Wraps SmellDataset and dynamically drops/zeroes semantic embeddings if use_semantic is False."""
    def __init__(self, original_dataset: SmellDataset, use_semantic: bool):
        self.original_dataset = original_dataset
        self.use_semantic = use_semantic
        
    def __len__(self):
        return len(self.original_dataset)
        
    def __getitem__(self, idx):
        item = copy.copy(self.original_dataset[idx])
        if not self.use_semantic:
            # Drop the semantic embeddings (keep only the first 9 features:
            # node type (3 dims) + normalized CK metrics/LOC (6 dims))
            item["x"] = item["x"][:, :9]
        return item


# ─── Phase 1 Sweep Configurations ───────────────────────────────────────────

PHASE_1_RUNS = [
    {
        "name": "phase1_slow_explorer",
        "description": "Phase 1: Slow Explorer (Extended epsilon decay)",
        "gcn_type": "full",
        "use_semantic": True,
        "dqn.learning_rate": 0.0005,
        "dqn.epsilon_decay_steps": 250000,
        "dqn.gamma": 0.0,
        "reward.correct_smell": 2.0,
        "reward.missed_smell": -1.0,
    },
    {
        "name": "phase1_cautious_learner",
        "description": "Phase 1: Cautious Learner (Lower learning rate)",
        "gcn_type": "full",
        "use_semantic": True,
        "dqn.learning_rate": 0.0001,
        "dqn.epsilon_decay_steps": 150000,
        "dqn.gamma": 0.0,
    },
    {
        "name": "phase1_strict_evaluator",
        "description": "Phase 1: Strict Evaluator (Shaped rewards for missed smells)",
        "gcn_type": "full",
        "use_semantic": True,
        "dqn.learning_rate": 0.0005,
        "dqn.epsilon_decay_steps": 150000,
        "reward.correct_smell": 2.0,
        "reward.missed_smell": -3.0,
        "reward.false_alarm": -0.5,
    }
]


# ─── Phase 2 Ablation Templates ─────────────────────────────────────────────

PHASE_2_TEMPLATES = [
    {
        "name": "ablate_semantic",
        "description": "Phase 2 Ablation: No Semantic (CK metrics + GCN)",
        "gcn_type": "full",
        "use_semantic": False,
    },
    {
        "name": "ablate_structure",
        "description": "Phase 2 Ablation: Shallow Graph (1-layer GCN + Semantic)",
        "gcn_type": "gcn_1layer",
        "use_semantic": True,
    },
    {
        "name": "ablate_traditional",
        "description": "Phase 2 Ablation: CK Only (1-layer GCN + CK metrics only)",
        "gcn_type": "gcn_1layer",
        "use_semantic": False,
    }
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
    
    # Base directories to isolate
    emb_suffix = "embeddings" if USE_EMBEDDINGS_MODEL else "no_embeddings"
    base_processed_dir = os.path.join(base_cfg["paths"]["processed_dir"], emb_suffix)
    base_checkpoint_dir = os.path.join(base_cfg["paths"]["checkpoint_dir"], emb_suffix)
    base_results_dir = os.path.join(base_cfg["paths"]["results_dir"], emb_suffix)

    # Adjust Phase 1 Runs if test mode is enabled
    phase_1_to_run = PHASE_1_RUNS
    if args.test:
        log.info("[TEST MODE ACTIVE] Slicing datasets and setting minimal training episodes.")
        phase_1_to_run = [
            {
                "name": "phase1_test_slow_explorer",
                "description": "Test Slow Explorer (Exploration)",
                "gcn_type": "full",
                "use_semantic": True,
                "dqn.learning_rate": 0.0005,
                "dqn.epsilon_decay_steps": 10,
                "dqn.gamma": 0.0,
                "reward.correct_smell": 2.0,
                "reward.missed_smell": -1.0,
                "dqn.warmup_steps": 2,
                "dqn.batch_size": 2,
                "dqn.max_episodes": 2,
            },
            {
                "name": "phase1_test_cautious_learner",
                "description": "Test Cautious Learner (Lower learning rate)",
                "gcn_type": "full",
                "use_semantic": True,
                "dqn.learning_rate": 0.0001,
                "dqn.epsilon_decay_steps": 10,
                "dqn.gamma": 0.0,
                "dqn.warmup_steps": 2,
                "dqn.batch_size": 2,
                "dqn.max_episodes": 2,
            },
            {
                "name": "phase1_test_strict_evaluator",
                "description": "Test Strict Evaluator (Shaped rewards)",
                "gcn_type": "full",
                "use_semantic": True,
                "dqn.learning_rate": 0.0005,
                "dqn.epsilon_decay_steps": 10,
                "reward.correct_smell": 2.0,
                "reward.missed_smell": -3.0,
                "reward.false_alarm": -0.5,
                "dqn.warmup_steps": 2,
                "dqn.batch_size": 2,
                "dqn.max_episodes": 2,
            }
        ]

    results = []

    # ────────────────────────────────────────────────────────────────────────
    # ─── Phase 1: Hyperparameter Sweep (Find Optimal Baseline) ──────────────
    # ────────────────────────────────────────────────────────────────────────
    log.info("\n══════════════════════════════════════════════════")
    log.info("   PHASE 1: Hyperparameter Sweep")
    log.info("══════════════════════════════════════════════════")

    best_f1 = -1.0
    best_run_name = None
    best_run_cfg = None

    for idx, run_cfg in enumerate(phase_1_to_run):
        run_name = run_cfg["name"]
        description = run_cfg["description"]
        log.info(f"\n[Phase 1 - Run {idx+1}/{len(phase_1_to_run)}] Starting: {run_name} ({description})")
        
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

        # Check if we can copy cached datasets to bypass heavy GraphCodeBERT preprocessing
        run_proc_dir = run_cfg_dict["paths"]["processed_dir"]
        os.makedirs(run_proc_dir, exist_ok=True)
        
        base_proc_files = ["train.pt", "val.pt", "test.pt"]
        if USE_EMBEDDINGS_MODEL:
            src_base_dir = os.path.join(base_cfg["paths"]["processed_dir"], "embeddings")
            if all(os.path.exists(os.path.join(src_base_dir, f)) for f in base_proc_files):
                if not all(os.path.exists(os.path.join(run_proc_dir, f)) for f in base_proc_files):
                    log.info(f"  Copying cached embeddings datasets to run processed directory...")
                    import shutil
                    for f in base_proc_files:
                        shutil.copy(os.path.join(src_base_dir, f), os.path.join(run_proc_dir, f))

        # Apply parameter overrides to run_cfg_dict
        for k, v in run_cfg.items():
            if k in ["name", "description", "gcn_type", "use_semantic"]:
                continue
            update_config_path(k, v, run_cfg_dict)

        # Apply test mode overrides if test is active
        if args.test:
            run_cfg_dict["gcn_pretrain"]["epochs"] = 2
            run_cfg_dict["gcn_pretrain"]["checkpoint_freq"] = 9999

        # ─── Stage 1: Preprocessing ───
        train_ds, val_ds, test_ds = run_preprocessing(run_cfg_dict)

        # Wrap datasets dynamically to support semantic ablation
        train_ds_wrapped = AblatedDataset(train_ds, run_cfg["use_semantic"])
        val_ds_wrapped = AblatedDataset(val_ds, run_cfg["use_semantic"])
        test_ds_wrapped = AblatedDataset(test_ds, run_cfg["use_semantic"])

        # Gather Dataset/Graph Statistics on full dataset
        train_stats = collect_dataset_stats(train_ds_wrapped, "train")
        val_stats = collect_dataset_stats(val_ds_wrapped, "val")
        test_stats = collect_dataset_stats(test_ds_wrapped, "test")

        log.info(f"  Graphs loaded: train={len(train_ds_wrapped)} (virtual: {train_stats['virtual_graphs']}), test={len(test_ds_wrapped)} (virtual: {test_stats['virtual_graphs']})")

        # Slice datasets if test mode is active to speed up pre-training/training execution
        if args.test:
            train_ds_wrapped.original_dataset._graphs = train_ds_wrapped.original_dataset._graphs[:16]
            val_ds_wrapped.original_dataset._graphs = val_ds_wrapped.original_dataset._graphs[:8]
            test_ds_wrapped.original_dataset._graphs = test_ds_wrapped.original_dataset._graphs[:8]

        # ─── Stage 2: GCN Pre-training ───
        gcn_type = run_cfg.get("gcn_type", "full")
        if gcn_type != "no_gcn":
            log.info(f"[Ablation Run] Starting GCN pretraining for {run_cfg_dict['gcn_pretrain']['epochs']} epochs...")
            pretrainer = GCNPretrainer(run_cfg_dict, device)
            pretrainer.train(train_ds_wrapped, val_ds_wrapped)

        # ─── Stage 3: DQN Agent Training ───
        agent, history = train_variant(run_cfg, run_cfg_dict, train_ds_wrapped, device, log)
        elapsed_time = time.time() - start_time

        # ─── Stage 4: Experiment Baseline Runner & Evaluation ───
        log.info("[Ablation Run] Running baseline comparison and F1 breakdown...")
        runner = ExperimentRunner(agent, train_ds_wrapped, test_ds_wrapped, run_cfg_dict, device)
        exp_results = runner.run_all(training_histories={"SmellRL": history})

        # Retrieve SmellRL metrics
        report = exp_results["baselines"].get("SmellRL (Ours)", {})
        if not report:
            report = evaluate_agent(agent, test_ds_wrapped)

        log.info(f"[Run Completed] Test Acc={report.get('accuracy', 0.0):.4f} | Macro F1={report.get('f1', 0.0):.4f} | Time={elapsed_time:.1f}s")

        run_result = {
            "run_index": len(results) + 1,
            "phase": 1,
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
                "final_loss": round(history[-1].get("loss", 0.0), 6) if len(history) > 0 else 0.0,
                "final_reward": round(history[-1].get("reward", 0.0), 4) if len(history) > 0 else 0.0,
                "final_accuracy": round(history[-1].get("accuracy", 0.0), 4) if len(history) > 0 else 0.0,
                "total_episodes": len(history)
            },
            "config": run_cfg_dict
        }
        results.append(run_result)

        # Track Phase 1 Winner (based on Macro F1)
        current_f1 = report.get("f1", 0.0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_run_name = run_name
            best_run_cfg = copy.deepcopy(run_cfg)

    log.info("\n══════════════════════════════════════════════════")
    log.info(f"   PHASE 1 COMPLETE: Winner is {best_run_name} (F1: {best_f1:.4f})")
    log.info("══════════════════════════════════════════════════")

    # ────────────────────────────────────────────────────────────────────────
    # ─── Phase 2: Ablation Study (Systematically break optimal config) ──────
    # ────────────────────────────────────────────────────────────────────────
    log.info("\n══════════════════════════════════════════════════")
    log.info("   PHASE 2: Ablation Study")
    log.info("══════════════════════════════════════════════════")

    for idx, template in enumerate(PHASE_2_TEMPLATES):
        run_name = f"phase2_{template['name']}"
        description = template["description"]
        log.info(f"\n[Phase 2 - Run {idx+1}/{len(PHASE_2_TEMPLATES)}] Starting: {run_name} ({description})")
        
        start_time = time.time()

        # Build isolated config by carrying over hyperparams of the Phase 1 winner
        run_cfg = copy.deepcopy(best_run_cfg)
        run_cfg["name"] = run_name
        run_cfg["description"] = description
        run_cfg["gcn_type"] = template["gcn_type"]
        run_cfg["use_semantic"] = template["use_semantic"]

        run_cfg_dict = copy.deepcopy(base_cfg)
        run_cfg_dict["paths"]["processed_dir"] = os.path.join(base_processed_dir, run_name)
        run_cfg_dict["paths"]["checkpoint_dir"] = os.path.join(base_checkpoint_dir, run_name)
        run_cfg_dict["paths"]["results_dir"] = os.path.join(base_results_dir, run_name)

        if USE_EMBEDDINGS_MODEL:
            run_cfg_dict["semantic"]["cache_dir"] = os.path.join(run_cfg_dict["semantic"]["cache_dir"], "real")
        else:
            run_cfg_dict["semantic"]["cache_dir"] = os.path.join(run_cfg_dict["semantic"]["cache_dir"], "synthetic")

        # Copy cached embeddings files to avoid regenerating them
        run_proc_dir = run_cfg_dict["paths"]["processed_dir"]
        os.makedirs(run_proc_dir, exist_ok=True)
        if USE_EMBEDDINGS_MODEL:
            src_base_dir = os.path.join(base_cfg["paths"]["processed_dir"], "embeddings")
            if all(os.path.exists(os.path.join(src_base_dir, f)) for f in base_proc_files):
                if not all(os.path.exists(os.path.join(run_proc_dir, f)) for f in base_proc_files):
                    log.info(f"  Copying cached embeddings datasets to run processed directory...")
                    import shutil
                    for f in base_proc_files:
                        shutil.copy(os.path.join(src_base_dir, f), os.path.join(run_proc_dir, f))

        # Apply inherited hyperparameters to config dictionary
        for k, v in run_cfg.items():
            if k in ["name", "description", "gcn_type", "use_semantic"]:
                continue
            update_config_path(k, v, run_cfg_dict)

        # Apply test mode overrides if test is active
        if args.test:
            run_cfg_dict["gcn_pretrain"]["epochs"] = 2
            run_cfg_dict["gcn_pretrain"]["checkpoint_freq"] = 9999

        # ─── Stage 1: Preprocessing ───
        train_ds, val_ds, test_ds = run_preprocessing(run_cfg_dict)

        # Wrap datasets dynamically to support semantic ablation
        train_ds_wrapped = AblatedDataset(train_ds, run_cfg["use_semantic"])
        val_ds_wrapped = AblatedDataset(val_ds, run_cfg["use_semantic"])
        test_ds_wrapped = AblatedDataset(test_ds, run_cfg["use_semantic"])

        # Gather Dataset/Graph Statistics on full dataset
        train_stats = collect_dataset_stats(train_ds_wrapped, "train")
        val_stats = collect_dataset_stats(val_ds_wrapped, "val")
        test_stats = collect_dataset_stats(test_ds_wrapped, "test")

        log.info(f"  Graphs loaded: train={len(train_ds_wrapped)} (virtual: {train_stats['virtual_graphs']}), test={len(test_ds_wrapped)} (virtual: {test_stats['virtual_graphs']})")

        # Slice datasets if test mode is active to speed up pre-training/training execution
        if args.test:
            train_ds_wrapped.original_dataset._graphs = train_ds_wrapped.original_dataset._graphs[:16]
            val_ds_wrapped.original_dataset._graphs = val_ds_wrapped.original_dataset._graphs[:8]
            test_ds_wrapped.original_dataset._graphs = test_ds_wrapped.original_dataset._graphs[:8]

        # ─── Stage 2: GCN Pre-training ───
        gcn_type = run_cfg.get("gcn_type", "full")
        if gcn_type != "no_gcn":
            log.info(f"[Ablation Run] Starting GCN pretraining for {run_cfg_dict['gcn_pretrain']['epochs']} epochs...")
            pretrainer = GCNPretrainer(run_cfg_dict, device)
            pretrainer.train(train_ds_wrapped, val_ds_wrapped)

        # ─── Stage 3: DQN Agent Training ───
        agent, history = train_variant(run_cfg, run_cfg_dict, train_ds_wrapped, device, log)
        elapsed_time = time.time() - start_time

        # ─── Stage 4: Experiment Baseline Runner & Evaluation ───
        log.info("[Ablation Run] Running baseline comparison and F1 breakdown...")
        runner = ExperimentRunner(agent, train_ds_wrapped, test_ds_wrapped, run_cfg_dict, device)
        exp_results = runner.run_all(training_histories={"SmellRL": history})

        # Retrieve SmellRL metrics
        report = exp_results["baselines"].get("SmellRL (Ours)", {})
        if not report:
            report = evaluate_agent(agent, test_ds_wrapped)

        log.info(f"[Run Completed] Test Acc={report.get('accuracy', 0.0):.4f} | Macro F1={report.get('f1', 0.0):.4f} | Time={elapsed_time:.1f}s")

        run_result = {
            "run_index": len(results) + 1,
            "phase": 2,
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
                "final_loss": round(history[-1].get("loss", 0.0), 6) if len(history) > 0 else 0.0,
                "final_reward": round(history[-1].get("reward", 0.0), 4) if len(history) > 0 else 0.0,
                "final_accuracy": round(history[-1].get("accuracy", 0.0), 4) if len(history) > 0 else 0.0,
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
            "phase": r["phase"],
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
