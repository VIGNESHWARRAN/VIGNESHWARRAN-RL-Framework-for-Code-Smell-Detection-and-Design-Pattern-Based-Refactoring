"""
Logging, checkpointing, and device utilities for SmellRL.
All training stages use these to ensure resume-ability and full traceability.
"""

import logging
import os
import glob
import json
import torch
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any


# ─────────────────────────── Logger ────────────────────────────────────────


def setup_logger(name: str, log_dir: str, level: int = logging.DEBUG) -> logging.Logger:
    """Create a logger that writes to both a timestamped file and stdout."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"[Logger] Initialized. Writing to: {log_file}")
    return logger


# ─────────────────────────── Device ────────────────────────────────────────


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(preference)
    return device


# ─────────────────────────── Checkpoints ───────────────────────────────────


class CheckpointManager:
    """
    Saves and loads training checkpoints.

    A checkpoint stores:
      - stage name (gcn_pretrain / smellrl_train)
      - episode / epoch number
      - model state dicts
      - optimizer state dicts
      - scalar training state (epsilon, best metrics, etc.)
      - full training history list

    The latest checkpoint is always named  <stage>_latest.pt
    Every `save_every` steps a numbered copy is also saved.
    """

    def __init__(self, checkpoint_dir: str, stage: str, logger: logging.Logger):
        self.dir = checkpoint_dir
        self.stage = stage
        self.log = logger
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _latest_path(self) -> str:
        return os.path.join(self.dir, f"{self.stage}_latest.pt")

    def _numbered_path(self, step: int) -> str:
        return os.path.join(self.dir, f"{self.stage}_step{step:05d}.pt")

    def save(
        self,
        step: int,
        models: Dict[str, torch.nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        history: list,
        save_numbered: bool = False,
    ) -> None:
        payload = {
            "stage": self.stage,
            "step": step,
            "state": state,
            "history": history,
            "models": {k: v.state_dict() for k, v in models.items()},
            "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
        }
        path = self._latest_path()
        torch.save(payload, path)
        self.log.info(f"[Checkpoint] Saved step={step} → {path}")

        if save_numbered:
            numbered = self._numbered_path(step)
            torch.save(payload, numbered)
            self.log.debug(f"[Checkpoint] Numbered copy → {numbered}")

    def load_latest(self) -> Optional[Dict]:
        path = self._latest_path()
        if not os.path.exists(path):
            self.log.info(f"[Checkpoint] No checkpoint found at {path}. Starting fresh.")
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.log.info(f"[Checkpoint] Loaded step={payload['step']} from {path}")
        return payload

    def restore(
        self,
        payload: Dict,
        models: Dict[str, torch.nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        device: torch.device,
    ) -> tuple:
        """Load weights into models/optimizers, return (step, state, history)."""
        for k, m in models.items():
            if k in payload["models"]:
                m.load_state_dict(payload["models"][k])
                m.to(device)
                self.log.debug(f"[Checkpoint] Restored model '{k}'")
            else:
                self.log.warning(f"[Checkpoint] Model key '{k}' not found in checkpoint")

        for k, opt in optimizers.items():
            if k in payload["optimizers"]:
                opt.load_state_dict(payload["optimizers"][k])
                self.log.debug(f"[Checkpoint] Restored optimizer '{k}'")

        return payload["step"], payload["state"], payload["history"]
