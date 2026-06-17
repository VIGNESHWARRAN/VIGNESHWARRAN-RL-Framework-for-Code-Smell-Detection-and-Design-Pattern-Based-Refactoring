"""
Run ablation studies (Section 7 of paper).

Ablation variants:
  1. Full SmellRL (loaded from checkpoint)
  2. No GCN — metric-vector-only state (node features averaged without message passing)
  3. Reward = ΔMI only        (alpha=1.0, others=0)
  4. Reward = ΔMI + I_smell   (no I_pattern)
  5. 1-layer GCN              (num_layers=1)

Usage:
  python scripts/run_ablations.py [--config config.yaml]

Results saved to data/results/ablations.csv and ablations.json.
"""

import argparse
import copy
import logging
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils      import setup_logger, get_device, CheckpointManager
from src.data       import run_preprocessing, SmellDataset
from src.models     import HierarchicalDQNAgent, GCNEncoder, GCNLayer
from src.training   import HierarchicalTrainer
from src.evaluation import ExperimentRunner
import torch.nn as nn
import torch.nn.functional as F


log = logging.getLogger("SmellRL.ablations")


# ── Identity GCN (no message passing — ablation 2) ─────────────────────────

class IdentityGCN(GCNEncoder):
    """Replaces graph convolution with a plain MLP on the class node only."""

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Only use class node (index 0)  — no message passing
        h = x[0:1]                      # [1, in_ch]
        h = F.relu(self.conv1.linear(h))
        h = self.conv2.linear(h)
        return h                        # [1, out_ch]


# ── Build and train an ablation variant ────────────────────────────────────

def build_ablation_agent(name: str, base_cfg: dict, device: torch.device) -> HierarchicalDQNAgent:
    cfg = copy.deepcopy(base_cfg)

    if name == "no_gcn":
        agent = HierarchicalDQNAgent(cfg, device)
        # Replace GCN with identity
        gcn_cfg = cfg["gcn"]
        new_gcn = IdentityGCN(gcn_cfg["node_feature_dim"], gcn_cfg["hidden_dim"],
                               gcn_cfg["output_dim"], gcn_cfg["dropout"])
        agent.gcn = new_gcn.to(device)
        return agent

    if name == "reward_mi_only":
        cfg["reward"] = {"alpha": 1.0, "beta": 0.0, "gamma_r": 0.0, "delta": 0.0, "lambda_r": 0.0}

    if name == "reward_mi_plus_smell":
        cfg["reward"] = {"alpha": 0.6, "beta": 0.0, "gamma_r": 0.4, "delta": 0.0, "lambda_r": 0.0}

    if name == "gcn_1layer":
        cfg["gcn"]["num_layers"] = 1   # trainer will use num_layers to build

    return HierarchicalDQNAgent(cfg, device)


def train_ablation(
    name: str,
    agent: HierarchicalDQNAgent,
    cfg: dict,
    train_ds: SmellDataset,
    val_ds:   SmellDataset,
    device:   torch.device,
) -> HierarchicalDQNAgent:
    """Train the ablation variant for a reduced number of episodes (50% of full)."""
    ablation_cfg = copy.deepcopy(cfg)
    ablation_cfg["training"]["max_episodes"] = max(50, cfg["training"]["max_episodes"] // 2)
    ablation_cfg["training"]["checkpoint_freq"] = 9999  # don't overwrite main checkpoint
    ablation_cfg["paths"]["checkpoint_dir"] = os.path.join(
        cfg["paths"]["checkpoint_dir"], f"ablation_{name}"
    )

    trainer = HierarchicalTrainer(agent, ablation_cfg, device)
    trainer.train(train_ds, val_ds)
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("device", "auto"))
    log    = setup_logger("SmellRL.ablations", cfg["paths"]["log_dir"])

    log.info("════════════════ Ablation Studies ════════════════")

    # Load processed datasets
    train_ds, val_ds, test_ds = run_preprocessing(cfg)

    # 1. Full SmellRL — load from main checkpoint
    log.info("[Ablation] Loading full SmellRL from checkpoint...")
    full_agent = HierarchicalDQNAgent(cfg, device)
    ckpt_mgr   = CheckpointManager(cfg["paths"]["checkpoint_dir"], "smellrl", log)
    payload    = ckpt_mgr.load_latest()
    if payload:
        models = {
            "gcn": full_agent.gcn, "q1": full_agent.q1, "q2": full_agent.q2,
            "q1_target": full_agent.q1_target, "q2_target": full_agent.q2_target,
        }
        ckpt_mgr.restore(payload, models, {}, device)

    ablation_variants = {"Full SmellRL": full_agent}

    # 2–5. Build and train ablation variants
    variant_names = {
        "no_gcn":               "No GCN (metric-vector only)",
        "reward_mi_only":       "Reward = ΔMI only",
        "reward_mi_plus_smell": "Reward = ΔMI + I_smell",
        "gcn_1layer":           "1-layer GCN",
    }

    for key, display_name in variant_names.items():
        log.info(f"[Ablation] Training variant: {display_name}")
        agent = build_ablation_agent(key, cfg, device)
        agent = train_ablation(key, agent, cfg, train_ds, val_ds, device)
        ablation_variants[display_name] = agent

    # Evaluate all variants
    runner = ExperimentRunner(full_agent, None, train_ds, test_ds, cfg, device)
    results = runner.run_ablations(ablation_variants)

    log.info("════════════════ Ablation Results ════════════════")
    for row in results:
        log.info(f"  {row['Variant']:40s} | F1={row['Phase-1 F1']:.4f} | Joint={row['Joint Acc']:.4f}")


if __name__ == "__main__":
    main()
