"""
SmellRL — Ablation Study Runner (v2: RGAT + Focal Loss architecture)

Research Methodology (two phases):

  Phase 1 — Architecture Ablation (answer: "What does each upgrade contribute?")
    Compares the new RGAT + Focal Loss system against degraded variants,
    each isolating one architectural contribution:
      A. Ours (Full)       : RGAT encoder  + Focal Loss          ← PRIMARY
      B. GCN + Focal Loss  : Legacy GCN    + Focal Loss          ← isolates encoder gain
      C. RGAT + CE Loss    : RGAT encoder  + CrossEntropy        ← isolates focal-loss gain
      D. GCN + CE Loss     : Legacy GCN    + CrossEntropy        ← old supervised baseline
      E. DQN (Contextual Bandit) : RGAT encoder + DQN reward     ← old RL baseline
      F. No Semantic       : RGAT + Focal, semantic dims zeroed  ← isolates semantic gain
      G. Severity Unfiltered : RGAT + Focal, all severities kept ← isolates data-filter gain

  Phase 2 — Hyperparameter Sweep (answer: "What γ / LR gives peak accuracy?")
    Uses the full architecture (RGAT + Focal) and sweeps key hyperparams:
      - focal_gamma in {1.0, 2.0, 3.0}
      - learning_rate in {1e-3, 3e-4, 1e-4}

Results are saved per-run in isolated directories and aggregated into a
single JSON + CSV in data/results/ablation/.

Usage:
  python scripts/run_ablations.py                  # full ablation suite
  python scripts/run_ablations.py --phase 1        # architecture ablation only
  python scripts/run_ablations.py --phase 2        # hyperparameter sweep only
  python scripts/run_ablations.py --test           # smoke test (2 epochs, 16 samples)
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
from src.data       import run_preprocessing, SmellDataset, SMELL_CLASSES, NUM_EDGE_TYPES
from src.models     import (
    RGATEncoder, GCNEncoder, SupervisedSmellDetector,
    SmellDetectionAgent, NodeFeatureFusion,
)
from src.training   import SupervisedTrainer, DQNTrainer, FocalLoss
from src.evaluation import classification_report_dict, ExperimentRunner


# ──────────────────────────── AblatedDataset ───────────────────────────────

class AblatedDataset(torch.utils.data.Dataset):
    """
    Wraps SmellDataset and optionally zeroes semantic embedding dims.

    NodeFeatureFusion always expects the full [N, 783] input; zeroing dims
    [15:783] rather than slicing ensures no shape mismatch while disabling
    the semantic branch (sem_proj(zeros) ≈ bias-only ≈ near-zero output).
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
            item["x"][:, NodeFeatureFusion.STRUCT_DIM:] = 0.0
        return item


# ──────────────────────────── Phase 1 — Architecture Ablations ─────────────

PHASE_1_RUNS = [
    # ── Primary model ────────────────────────────────────────────────────────
    {
        "name":          "full_rgat_focal",
        "description":   "Ours (Full): RGAT encoder + Focal Loss (γ=2)",
        "encoder_type":  "rgat",
        "loss":          "focal",
        "use_semantic":  True,
        "filter_minor":  True,
    },
    # ── Ablate the encoder (GCN instead of RGAT) ─────────────────────────────
    {
        "name":          "gcn_focal",
        "description":   "Ablation: Legacy GCN + Focal Loss — isolates RGAT gain",
        "encoder_type":  "gcn",
        "loss":          "focal",
        "use_semantic":  True,
        "filter_minor":  True,
    },
    # ── Ablate the loss (CrossEntropy instead of Focal) ───────────────────────
    {
        "name":          "rgat_ce",
        "description":   "Ablation: RGAT + CrossEntropy — isolates Focal Loss gain",
        "encoder_type":  "rgat",
        "loss":          "cross_entropy",
        "use_semantic":  True,
        "filter_minor":  True,
    },
    # ── Ablate both encoder and loss (old supervised baseline) ────────────────
    {
        "name":          "gcn_ce",
        "description":   "Ablation: Legacy GCN + CrossEntropy — old supervised baseline",
        "encoder_type":  "gcn",
        "loss":          "cross_entropy",
        "use_semantic":  True,
        "filter_minor":  True,
    },
    # ── Ablate training paradigm (DQN instead of supervised) ─────────────────
    {
        "name":          "rgat_dqn",
        "description":   "Ablation: RGAT encoder + DQN contextual bandit — old RL baseline",
        "encoder_type":  "rgat",
        "loss":          "dqn",        # special flag → use DQNTrainer
        "use_semantic":  True,
        "filter_minor":  True,
    },
    # ── Ablate semantic embeddings ────────────────────────────────────────────
    {
        "name":          "rgat_focal_no_semantic",
        "description":   "Ablation: RGAT + Focal, semantic dims zeroed — isolates semantic gain",
        "encoder_type":  "rgat",
        "loss":          "focal",
        "use_semantic":  False,
        "filter_minor":  True,
    },
    # ── Ablate data-centric filter ────────────────────────────────────────────
    {
        "name":          "rgat_focal_all_severity",
        "description":   "Ablation: RGAT + Focal, all severities kept — isolates data-filter gain",
        "encoder_type":  "rgat",
        "loss":          "focal",
        "use_semantic":  True,
        "filter_minor":  False,     # keeps minor-severity noise
    },
]


# ──────────────────────────── Phase 2 — Hyperparameter Sweep ───────────────

PHASE_2_RUNS = []
for _gamma in [1.0, 2.0, 3.0]:
    for _lr in [1e-3, 3e-4, 1e-4]:
        PHASE_2_RUNS.append({
            "name":          f"sweep_g{_gamma}_lr{_lr:.0e}".replace("-", "m"),
            "description":   f"Sweep: RGAT + Focal (γ={_gamma}, lr={_lr:.0e})",
            "encoder_type":  "rgat",
            "loss":          "focal",
            "use_semantic":  True,
            "filter_minor":  True,
            "focal_gamma":   _gamma,
            "learning_rate": _lr,
        })


# ──────────────────────────── Helpers ──────────────────────────────────────

def _make_synthetic_ds(n: int, feature_dim: int = 783, n_classes: int = 5) -> SmellDataset:
    """
    Build a tiny in-memory SmellDataset of `n` random graphs.

    Used ONLY in --test mode so the ablation smoke-test never touches the
    MLCQ CSV pipeline or the GraphCodeBERT embedder. Each graph has:
      - 4 nodes (1 class + 3 methods)
      - 6 edges (CONTAINS×2 + CALLS×2 + ACCESSES_FIELD×2)
      - Labels round-robined over SMELL_CLASSES
    """
    import random as _r
    graphs = []
    for i in range(n):
        x  = torch.randn(4, feature_dim)
        ei = torch.tensor([[0,1,0,2,1,2],[1,0,2,0,2,1]], dtype=torch.long)
        et = torch.tensor([0,0,1,1,2,2], dtype=torch.long)
        y  = torch.tensor(i % n_classes, dtype=torch.long)
        graphs.append({"x": x, "edge_index": ei, "edge_type": et, "y": y})
    ds = SmellDataset.__new__(SmellDataset)
    ds._graphs = graphs
    return ds

def collect_dataset_stats(dataset) -> dict:
    """Per-class sample counts for a dataset split."""
    class_counts = {s: 0 for s in SMELL_CLASSES}
    for item in dataset:
        class_counts[SMELL_CLASSES[int(item["y"].item())]] += 1
    return {"total_graphs": len(dataset), "class_distribution": class_counts}


@torch.no_grad()
def evaluate_detector(detector, test_ds) -> dict:
    """Standard accuracy/F1 report for any select_action-compatible detector."""
    detector.eval()
    y_true, y_pred = [], []
    for item in test_ds:
        y_true.append(int(item["y"].item()))
        action, _ = detector.select_action(item, epsilon=0.0)
        y_pred.append(action)
    detector.train()
    return classification_report_dict(y_true, y_pred, SMELL_CLASSES)


def build_encoder(encoder_type: str, feature_dim: int, cfg: dict, device: torch.device):
    """Instantiate RGAT or legacy GCN encoder from config."""
    gcn_cfg = cfg["gcn"]
    if encoder_type == "rgat":
        return RGATEncoder(
            in_ch=feature_dim,
            hidden=gcn_cfg["hidden_dim"],
            out_ch=gcn_cfg["output_dim"],
            num_edge_types=gcn_cfg.get("num_edge_types", NUM_EDGE_TYPES),
            num_heads=gcn_cfg.get("num_heads", 4),
            dropout=gcn_cfg["dropout"],
        ).to(device)
    else:
        return GCNEncoder(
            in_ch=feature_dim,
            hidden=gcn_cfg["hidden_dim"],
            out_ch=gcn_cfg["output_dim"],
            dropout=gcn_cfg["dropout"],
        ).to(device)


# ──────────────────────────── Single-run executor ──────────────────────────

def run_single(
    run_spec:    dict,
    base_cfg:    dict,
    base_dir:    str,
    device:      torch.device,
    log:         logging.Logger,
    test_mode:   bool = False,
) -> dict:
    """
    Execute one ablation run end-to-end.

    Returns a result dict with metrics, timing, and dataset stats.
    """
    run_name    = run_spec["name"]
    description = run_spec["description"]
    log.info(f"\n{'─'*60}")
    log.info(f"  RUN: {run_name}")
    log.info(f"  {description}")
    log.info(f"{'─'*60}")

    start_time = time.time()

    # ── Build isolated config ────────────────────────────────────────────────
    run_cfg = copy.deepcopy(base_cfg)

    # ── Processed-data directory strategy ────────────────────────────────────
    # The graph .pt files only depend on the severity filter, NOT on the encoder
    # type, loss function, or learning rate.  Pointing all same-filter runs at
    # the same processed_dir means preprocessing runs exactly once (or reuses
    # what the main pipeline already built) instead of once per ablation.
    #
    #   filter_minor=True  (default) → reuse main pipeline data/processed/
    #   filter_minor=False (severity ablation) → isolated dir so the two datasets
    #                                             don't overwrite each other.
    if run_spec.get("filter_minor", True):
        run_cfg["paths"]["processed_dir"] = base_cfg["paths"]["processed_dir"]
    else:
        run_cfg["paths"]["processed_dir"] = os.path.join(
            base_dir, "processed", run_name
        )

    run_cfg["paths"]["checkpoint_dir"] = os.path.join(base_dir, "checkpoints", run_name)
    run_cfg["paths"]["results_dir"]    = os.path.join(base_dir, "results",    run_name)

    # Shared embedding cache (all runs can reuse the same GraphCodeBERT cache)
    run_cfg["semantic"]["cache_dir"] = base_cfg["semantic"]["cache_dir"]

    # Apply per-run hyperparameter overrides
    if "focal_gamma" in run_spec:
        run_cfg.setdefault("supervised_train", {})["focal_gamma"] = run_spec["focal_gamma"]
        run_cfg.setdefault("gcn_pretrain", {})["focal_gamma"] = run_spec["focal_gamma"]
    if "learning_rate" in run_spec:
        run_cfg.setdefault("supervised_train", {})["learning_rate"] = run_spec["learning_rate"]

    # Apply encoder_type override
    run_cfg["gcn"]["encoder_type"] = run_spec["encoder_type"]

    # Apply severity filter override
    run_cfg["dataset"]["filter_minor_severity"] = run_spec.get("filter_minor", True)

    # Test-mode: shrink epochs and dataset drastically
    if test_mode:
        run_cfg.setdefault("supervised_train", {}).update({
            "epochs": 2, "eval_freq": 1, "early_stopping_patience": 99,
            "checkpoint_freq": 9999, "batch_size": 4,
        })
        run_cfg.setdefault("gcn_pretrain", {}).update({
            "epochs": 2, "checkpoint_freq": 9999,
        })
        run_cfg.setdefault("dqn", {}).update({
            "max_episodes": 2, "warmup_steps": 2, "batch_size": 2,
        })

    for p in ["checkpoint_dir", "results_dir"]:
        os.makedirs(run_cfg["paths"][p], exist_ok=True)
    os.makedirs(run_cfg["paths"]["processed_dir"], exist_ok=True)

    # ── Preprocessing / synthetic data ───────────────────────────────────────
    # In --test mode, skip the entire CSV + GraphCodeBERT + AST pipeline and
    # use tiny random tensors instead. This makes the smoke-test instant and
    # fully independent of data availability.
    if test_mode:
        log.info("  [TEST MODE] Using synthetic mini-dataset (skipping CSV/AST pipeline)")
        train_ds_raw = _make_synthetic_ds(20)
        val_ds_raw   = _make_synthetic_ds(10)
        test_ds_raw  = _make_synthetic_ds(10)
    else:
        train_ds_raw, val_ds_raw, test_ds_raw = run_preprocessing(run_cfg)

    # Apply semantic ablation wrapper if needed
    use_sem = run_spec.get("use_semantic", True)
    train_ds = AblatedDataset(train_ds_raw, use_sem)
    val_ds   = AblatedDataset(val_ds_raw,   use_sem)
    test_ds  = AblatedDataset(test_ds_raw,  use_sem)

    if test_mode:
        train_ds.original_dataset._graphs = train_ds.original_dataset._graphs[:16]
        val_ds.original_dataset._graphs   = val_ds.original_dataset._graphs[:8]
        test_ds.original_dataset._graphs  = test_ds.original_dataset._graphs[:8]

    train_stats = collect_dataset_stats(train_ds)
    val_stats   = collect_dataset_stats(val_ds)
    test_stats  = collect_dataset_stats(test_ds)

    log.info(
        f"  Dataset: train={train_stats['total_graphs']}, "
        f"val={val_stats['total_graphs']}, test={test_stats['total_graphs']}"
    )

    feature_dim = train_ds[0]["x"].shape[1] if len(train_ds) > 0 else 783

    # ── Train ─────────────────────────────────────────────────────────────────
    loss_type    = run_spec.get("loss", "focal")
    encoder_type = run_spec["encoder_type"]

    if loss_type == "dqn":
        # ── DQN ablation path ────────────────────────────────────────────────
        log.info("  Training path: DQN contextual bandit (ablation)")
        agent   = SmellDetectionAgent(run_cfg, feature_dim, device)
        trainer = DQNTrainer(agent, run_cfg, device)
        history = trainer.train(train_ds, val_ds)
        detector = agent

    else:
        # ── Supervised path (primary + all non-DQN ablations) ────────────────
        log.info(f"  Training path: Supervised ({encoder_type} + {loss_type})")

        # Temporarily inject loss type into config for SupervisedTrainer
        run_cfg.setdefault("gcn_pretrain", {})["loss"] = loss_type

        # Build encoder and classifier manually to support GCN override
        encoder    = build_encoder(encoder_type, feature_dim, run_cfg, device)
        n_classes  = run_cfg["smells"]["n_classes"]
        out_dim    = run_cfg["gcn"]["output_dim"]
        classifier = nn.Linear(out_dim, n_classes).to(device)

        # Compute class weights from training data
        counts = {}
        for item in train_ds:
            y = int(item["y"].item())
            counts[y] = counts.get(y, 0) + 1
        total   = sum(counts.values())
        alpha   = torch.ones(n_classes)
        for cls_idx, count in counts.items():
            alpha[cls_idx] = total / (n_classes * count) if count > 0 else 1.0
        alpha = alpha.to(device)

        if loss_type == "focal":
            gamma     = run_spec.get("focal_gamma",
                        run_cfg.get("supervised_train", {}).get("focal_gamma", 2.0))
            criterion = FocalLoss(gamma=gamma, alpha=alpha)
            log.info(f"  Loss: FocalLoss (γ={gamma})")
        else:
            criterion = nn.CrossEntropyLoss(weight=alpha)
            log.info("  Loss: CrossEntropyLoss (class-weighted)")

        sup_cfg  = run_cfg.get("supervised_train", {})
        epochs   = sup_cfg.get("epochs", 150)
        batch_sz = sup_cfg.get("batch_size", 32)
        lr       = run_spec.get("learning_rate", sup_cfg.get("learning_rate", 3e-4))
        wd       = sup_cfg.get("weight_decay", 1e-5)
        patience = sup_cfg.get("early_stopping_patience", 20)
        eval_freq = sup_cfg.get("eval_freq", 10)

        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(classifier.parameters()),
            lr=lr, weight_decay=wd,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0.0
        best_state   = None
        pat_counter  = 0
        history      = []

        import random
        for epoch in range(epochs):
            encoder.train(); classifier.train()
            indices    = list(range(len(train_ds)))
            random.shuffle(indices)
            total_loss = 0.0; correct = 0; batches = 0

            for start in range(0, len(indices), batch_sz):
                batch_idx = indices[start: start + batch_sz]
                optimizer.zero_grad()
                all_logits, all_labels = [], []
                for i in batch_idx:
                    item       = train_ds[i]
                    x          = item["x"].to(device)
                    edge_index = item["edge_index"].to(device)
                    y          = item["y"].to(device)

                    if encoder_type == "rgat":
                        emb = encoder(x, edge_index, item["edge_type"].to(device))
                    else:
                        emb = encoder(x, edge_index)

                    all_logits.append(classifier(emb))
                    all_labels.append(y.unsqueeze(0))

                bl = torch.cat(all_logits); tl = torch.cat(all_labels)
                loss = criterion(bl, tl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(classifier.parameters()), 1.0
                )
                optimizer.step()
                total_loss += loss.item()
                correct    += (bl.argmax(1) == tl).sum().item()
                batches    += 1

            scheduler.step()
            train_acc = correct / max(1, len(train_ds))
            stats = {"epoch": epoch+1, "train_loss": total_loss/max(1,batches), "train_accuracy": train_acc}

            if (epoch + 1) % eval_freq == 0:
                encoder.eval(); classifier.eval()
                val_correct = 0
                with torch.no_grad():
                    for item in val_ds:
                        x = item["x"].to(device); ei = item["edge_index"].to(device)
                        yy = item["y"].to(device)
                        if encoder_type == "rgat":
                            emb = encoder(x, ei, item["edge_type"].to(device))
                        else:
                            emb = encoder(x, ei)
                        if classifier(emb).argmax(1).item() == yy.item():
                            val_correct += 1
                val_acc = val_correct / max(1, len(val_ds))
                stats["val_accuracy"] = val_acc
                log.info(f"  Ep {epoch+1:3d}/{epochs} | loss={stats['train_loss']:.4f} | "
                         f"acc={train_acc:.4f} | val_acc={val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state   = {
                        "encoder":    copy.deepcopy(encoder.state_dict()),
                        "classifier": copy.deepcopy(classifier.state_dict()),
                    }
                    pat_counter = 0
                else:
                    pat_counter += 1
                    if pat_counter >= patience:
                        log.info(f"  Early stopping at epoch {epoch+1}")
                        history.append(stats); break
            history.append(stats)

        if best_state:
            encoder.load_state_dict(best_state["encoder"])
            classifier.load_state_dict(best_state["classifier"])
            log.info(f"  Restored best model (val_acc={best_val_acc:.4f})")

        detector = SupervisedSmellDetector(encoder, classifier, n_classes, device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    report  = evaluate_detector(detector, test_ds)

    log.info(
        f"  RESULT: Acc={report['accuracy']:.4f} | "
        f"Macro-F1={report['f1']:.4f} | "
        f"P={report['precision']:.4f} | R={report['recall']:.4f} | "
        f"Time={elapsed:.0f}s"
    )
    for smell in SMELL_CLASSES:
        pc = report["per_class"].get(smell, {})
        log.info(f"    {smell:15s}: P={pc.get('p',0):.3f}  R={pc.get('r',0):.3f}  F1={pc.get('f1',0):.3f}")

    # Save per-run results CSV
    per_run_path = os.path.join(run_cfg["paths"]["results_dir"], "ablation_metrics.csv")
    pd.DataFrame([{
        "run_name": run_name,
        "accuracy": report["accuracy"],
        "macro_f1": report["f1"],
        "macro_precision": report["precision"],
        "macro_recall": report["recall"],
        **{f"f1_{s}": report["per_class"].get(s, {}).get("f1", 0.0) for s in SMELL_CLASSES},
    }]).to_csv(per_run_path, index=False)

    return {
        "run_name":    run_name,
        "description": description,
        "elapsed_seconds": round(elapsed, 1),
        "hyperparameters": {
            k: v for k, v in run_spec.items()
            if k not in ["name", "description"]
        },
        "dataset_statistics": {
            "train": train_stats, "val": val_stats, "test": test_stats,
        },
        "metrics": {
            "accuracy":        round(report["accuracy"], 4),
            "macro_f1":        round(report["f1"], 4),
            "macro_precision": round(report["precision"], 4),
            "macro_recall":    round(report["recall"], 4),
            "per_class_f1": {
                s: round(report["per_class"].get(s, {}).get("f1", 0.0), 4)
                for s in SMELL_CLASSES
            },
        },
        "training_summary": {
            "total_epochs": len(history),
            "final_loss":   round(history[-1].get("train_loss", history[-1].get("loss", 0.0)), 6)
                            if history else 0.0,
            "best_val_acc": round(best_val_acc if loss_type != "dqn" else 0.0, 4),
        },
    }


# ──────────────────────────── Main ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SmellRL Ablation Studies (v2)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=None,
        help="Run only phase 1 (architecture ablations) or phase 2 (hyperparam sweep). "
             "Default: run both."
    )
    parser.add_argument(
        "--run", type=str, default=None,
        help="Run a single ablation by name (e.g. --run gcn_focal). "
             "Overrides --phase."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Smoke-test mode: 2 epochs, 16 training samples, skips heavy runs."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    device  = get_device(base_cfg.get("device", "auto"))
    log     = setup_logger("SmellRL.ablations", base_cfg["paths"]["log_dir"])
    base_dir = os.path.join(
        os.path.dirname(base_cfg["paths"]["results_dir"]),
        "ablations",
    )
    os.makedirs(base_dir, exist_ok=True)

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║        SmellRL Ablation Suite  (v2)             ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info(f"Device : {device}")
    log.info(f"Output : {base_dir}")
    if args.test:
        log.info("  *** TEST MODE: minimal epochs, tiny dataset ***")

    all_results = []

    # ── Select runs ──────────────────────────────────────────────────────────
    if args.run:
        all_specs = PHASE_1_RUNS + PHASE_2_RUNS
        specs = [s for s in all_specs if s["name"] == args.run]
        if not specs:
            log.error(f"No run named '{args.run}'. "
                      f"Available: {[s['name'] for s in all_specs]}")
            sys.exit(1)
    elif args.phase == 1:
        specs = PHASE_1_RUNS
    elif args.phase == 2:
        specs = PHASE_2_RUNS
    else:
        specs = PHASE_1_RUNS + PHASE_2_RUNS

    log.info(f"\nWill execute {len(specs)} run(s):\n")
    for i, s in enumerate(specs, 1):
        log.info(f"  {i:2d}. {s['name']:40s}  {s['description']}")
    log.info("")

    # ── Execute each run ─────────────────────────────────────────────────────
    for i, spec in enumerate(specs, 1):
        log.info(f"\n[{i}/{len(specs)}] Starting: {spec['name']}")
        try:
            result = run_single(spec, base_cfg, base_dir, device, log, args.test)
            all_results.append(result)
        except Exception as e:
            log.error(f"  FAILED: {spec['name']} — {e}", exc_info=True)
            all_results.append({
                "run_name": spec["name"],
                "description": spec["description"],
                "error": str(e),
                "metrics": {},
            })

    # ── Aggregate results ─────────────────────────────────────────────────────
    suffix = "_test" if args.test else ""
    json_path = os.path.join(base_dir, f"ablation_results{suffix}.json")
    csv_path  = os.path.join(base_dir, f"ablation_results{suffix}.csv")

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSaved JSON → {json_path}")

    csv_rows = []
    for r in all_results:
        if "error" in r:
            csv_rows.append({"run_name": r["run_name"], "error": r["error"]})
            continue
        row = {
            "run_name":        r["run_name"],
            "description":     r["description"],
            "elapsed_seconds": r["elapsed_seconds"],
            "accuracy":        r["metrics"].get("accuracy", 0.0),
            "macro_f1":        r["metrics"].get("macro_f1", 0.0),
            "macro_precision": r["metrics"].get("macro_precision", 0.0),
            "macro_recall":    r["metrics"].get("macro_recall", 0.0),
            "total_epochs":    r["training_summary"].get("total_epochs", 0),
            "best_val_acc":    r["training_summary"].get("best_val_acc", 0.0),
            **{f"f1_{s}": r["metrics"].get("per_class_f1", {}).get(s, 0.0) for s in SMELL_CLASSES},
            **{f"hp_{k}": v for k, v in r.get("hyperparameters", {}).items()},
        }
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    log.info(f"Saved CSV  → {csv_path}")

    # ── Print leaderboard ─────────────────────────────────────────────────────
    log.info("\n" + "═"*60)
    log.info("  ABLATION LEADERBOARD  (sorted by Accuracy)")
    log.info("═"*60)
    ranked = sorted(
        [r for r in all_results if "error" not in r],
        key=lambda r: r["metrics"].get("accuracy", 0.0),
        reverse=True,
    )
    log.info(f"  {'Run':<40} {'Acc':>6}  {'F1':>6}")
    log.info(f"  {'-'*40} {'------':>6}  {'------':>6}")
    for r in ranked:
        log.info(
            f"  {r['run_name']:<40} "
            f"{r['metrics'].get('accuracy', 0.0):>6.4f}  "
            f"{r['metrics'].get('macro_f1', 0.0):>6.4f}"
        )
    log.info("═"*60)
    log.info("\nDone.")


if __name__ == "__main__":
    main()
