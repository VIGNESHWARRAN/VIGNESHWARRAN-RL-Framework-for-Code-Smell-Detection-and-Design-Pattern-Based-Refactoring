"""
SmellRL v3 — main entry point.

Usage:
  python main.py                          # full pipeline (preprocess, pretrain, train, experiment)
  python main.py --stage preprocess       # data preprocessing only
  python main.py --stage pretrain         # GCN pre-training warm-start only
  python main.py --stage train            # RL training (PRIMARY)
  python main.py --stage experiment       # evaluation against baselines
"""

import argparse
import logging
import os
import sys
import yaml

# ──────────────────────────── Config ───────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# ──────────────────────────── Stage implementations ────────────────────────

def stage_preprocess(cfg: dict, log: logging.Logger) -> tuple:
    from src.data import run_preprocessing
    log.info("══════════════ STAGE: preprocess ══════════════")
    train_ds, val_ds, test_ds = run_preprocessing(cfg)
    log.info("Preprocessing complete.")
    return train_ds, val_ds, test_ds


def stage_pretrain(cfg: dict, train_ds, val_ds, device, log: logging.Logger):
    from src.training import GCNPretrainer
    log.info("══════════════ STAGE: GCN/RGAT pre-training (warm-start) ══════════════")
    pretrainer = GCNPretrainer(cfg, device)
    enc = pretrainer.train(train_ds, val_ds)
    log.info("Encoder pre-training complete.")
    return enc


def stage_train(cfg: dict, train_ds, val_ds, device, log: logging.Logger) -> tuple:
    from src.models import SmellDetectionAgent
    from src.training import DQNTrainer
    from src.utils import CheckpointManager
    log.info("══════════════ STAGE: RL DQN training (PRIMARY) ══════════════")
    feature_dim = train_ds[0]["x"].shape[1] if len(train_ds) > 0 else 783
    log.info(f"Detected node feature dimension: {feature_dim}")

    agent = SmellDetectionAgent(cfg, feature_dim, device)

    # NO fallback. Checkpoint MUST exist.
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    gcn_ckpt = CheckpointManager(ckpt_dir, "gcn_pretrain", log)
    payload  = gcn_ckpt.load_latest()
    if payload and "models" in payload and "gcn" in payload["models"]:
        agent.gcn.load_state_dict(payload["models"]["gcn"])
        log.info("Successfully loaded pre-trained encoder weights from gcn_pretrain checkpoint into DQN agent.")
    else:
        raise FileNotFoundError(
            f"No pre-trained encoder checkpoint found in {ckpt_dir} for gcn_pretrain. "
            "Please run 'python main.py --stage pretrain' first before training."
        )

    trainer = DQNTrainer(agent, cfg, device)
    history = trainer.train(train_ds, val_ds)
    log.info("RL DQN training complete.")
    return agent, history


def stage_experiment(
    cfg:        dict,
    agent,
    train_ds,
    test_ds,
    device,
    log:        logging.Logger,
    training_histories: dict = None,
):
    from src.evaluation import ExperimentRunner
    log.info("══════════════ STAGE: run experiments ══════════════")
    runner  = ExperimentRunner(agent, train_ds, test_ds, cfg, device)
    results = runner.run_all(training_histories)
    log.info("All experiments complete. Results saved to data/results/")
    return results


# ──────────────────────────── Full pipeline ────────────────────────────────

def run_all(cfg: dict, device, log: logging.Logger):
    """Runs all stages in order."""
    # 1. Preprocess
    train_ds, val_ds, test_ds = stage_preprocess(cfg, log)

    # 2. Supervised pre-training
    stage_pretrain(cfg, train_ds, val_ds, device, log)

    # 3. RL training
    agent, history = stage_train(cfg, train_ds, val_ds, device, log)

    # 4. Experiments
    histories = {"SmellRL (DQN)": history}
    stage_experiment(cfg, agent, train_ds, test_ds, device, log, histories)


# ──────────────────────────── Main entry ───────────────────────────────────

def main():
    cfg = load_config("config.yaml")
    
    from src.utils import setup_logger, get_device, CheckpointManager
    from src.models import SmellDetectionAgent
    
    log = setup_logger(cfg["paths"]["log_dir"], "SmellRL")
    
    parser = argparse.ArgumentParser(description="SmellRL v3 entrypoint")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["preprocess", "pretrain", "train", "experiment"],
        help="Run a specific stage only. If omitted, runs the entire pipeline."
    )
    args = parser.parse_args()

    device = get_device(cfg.get("device", "auto"))
    log.info(f"Active Device: {device}")

    # Ensure processed directory exists
    os.makedirs(cfg["paths"]["processed_dir"], exist_ok=True)

    if args.stage == "preprocess":
        stage_preprocess(cfg, log)
    elif args.stage == "pretrain":
        train_ds, val_ds, _ = stage_preprocess(cfg, log)
        stage_pretrain(cfg, train_ds, val_ds, device, log)
    elif args.stage == "train":
        train_ds, val_ds, _ = stage_preprocess(cfg, log)
        stage_train(cfg, train_ds, val_ds, device, log)
    elif args.stage == "experiment":
        train_ds, val_ds, test_ds = stage_preprocess(cfg, log)
        # Load trained agent checkpoint
        feature_dim = train_ds[0]["x"].shape[1] if len(train_ds) > 0 else 783
        agent = SmellDetectionAgent(cfg, feature_dim, device)
        
        dqn_ckpt = CheckpointManager(cfg["paths"]["checkpoint_dir"], "smellrl_dqn", log)
        payload = dqn_ckpt.load_latest()
        if not payload:
            raise FileNotFoundError("No trained DQN checkpoint found. Run training stage first.")
        agent.load_state_dict(payload["models"]["agent"])
        log.info("Loaded trained SmellRL agent from checkpoint.")
        
        stage_experiment(cfg, agent, train_ds, test_ds, device, log)
    else:
        run_all(cfg, device, log)

if __name__ == "__main__":
    main()