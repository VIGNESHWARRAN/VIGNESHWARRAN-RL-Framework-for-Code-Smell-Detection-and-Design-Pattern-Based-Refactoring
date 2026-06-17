"""
Training pipeline for SmellRL.

ReplayBuffer      — uniform experience replay
RewardFunction    — multi-component reward (Section 4.4 of paper)
GCNPretrainer     — supervised pre-training of the GCN encoder
HierarchicalTrainer — full RL training loop with resume support
FlatDQNTrainer    — trains the flat DQN baseline
"""

import os
import logging
import random
import time
import json
import math
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from src.data    import SMELL_TO_PATTERNS, SMELL_CLASSES, PATTERN_CLASSES, SmellDataset
from src.models  import HierarchicalDQNAgent, FlatDQNAgent, GCNClassifier, GCNEncoder
from src.utils   import CheckpointManager

logger = logging.getLogger("SmellRL.training")

# ──────────────────────────── Replay Buffer ────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, transition: dict):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[dict]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────── Reward Function ──────────────────────────────

# Expected metric deltas after applying each (smell, pattern) combination.
# Format: (ΔMI, ΔCBO, ΔCC)  — ΔMI and -ΔCBO are positive = good, ΔCC small = good
_PATTERN_DELTAS: Dict[Tuple[int,int], Tuple[float,float,float]] = {
    (0, 2): (18.0, -3.0, 1.0),   # GodClass    + Facade
    (0, 3): (15.0, -2.5, 1.5),   # GodClass    + Mediator
    (1, 0): (12.0, -2.0, 0.5),   # FeatureEnvy + Strategy
    (1, 4): (10.0, -1.5, 0.5),   # FeatureEnvy + ExtractClass
    (2, 4): (14.0, -1.0, 0.8),   # LongMethod  + ExtractClass
    (2, 5): ( 8.0, -0.5, 0.0),   # LongMethod  + None
    (3, 1): (11.0, -1.5, 0.3),   # DataClass   + Observer
    (3, 5): ( 6.0, -0.5, 0.0),   # DataClass   + None
    (4, 5): ( 2.0, -0.1, 0.0),   # NoSmell     + None
}
_DEFAULT_DELTA = (3.0, -0.5, 0.5)   # wrong pattern applied


def simulate_metric_delta(
    smell_idx: int, pattern_idx: int, noise_std: float = 0.5
) -> Tuple[float, float, float]:
    """Return (ΔMI, ΔCBO, ΔCC) for a (smell, pattern) pair with Gaussian noise."""
    base = _PATTERN_DELTAS.get((smell_idx, pattern_idx), _DEFAULT_DELTA)
    rng  = np.random
    return (
        base[0] + rng.normal(0, noise_std),
        base[1] + rng.normal(0, noise_std * 0.3),
        base[2] + rng.normal(0, noise_std * 0.2),
    )


class RewardFunction:
    """
    R = α·ΔMI + β·(−ΔCBO) + γ·I_smell + δ·I_pattern − λ·ΔCC

    Normalise each component so they're on comparable scales.
    """

    def __init__(self, rcfg: dict):
        self.alpha   = rcfg["alpha"]
        self.beta    = rcfg["beta"]
        self.gamma_r = rcfg["gamma_r"]
        self.delta   = rcfg["delta"]
        self.lambda_r= rcfg["lambda_r"]

    def __call__(
        self,
        true_smell: int,
        pred_smell: int,
        pred_pattern: int,
        noise_std: float = 0.5,
    ) -> Tuple[float, Dict]:
        dmi, dcbo, dcc = simulate_metric_delta(true_smell, pred_pattern, noise_std)

        i_smell   = float(pred_smell == true_smell)
        i_pattern = float(pred_pattern in SMELL_TO_PATTERNS.get(true_smell, []))

        # normalise continuous deltas to [0,1] approximately
        mi_norm  = dmi  / 20.0
        cbo_norm = (-dcbo) / 5.0   # larger reduction = better
        cc_pen   = dcc  / 5.0

        reward = (
            self.alpha   * mi_norm
            + self.beta  * cbo_norm
            + self.gamma_r * i_smell
            + self.delta * i_pattern
            - self.lambda_r * cc_pen
        )

        components = {
            "dmi": dmi, "dcbo": dcbo, "dcc": dcc,
            "i_smell": i_smell, "i_pattern": i_pattern,
            "reward": reward,
        }
        return reward, components


# ──────────────────────────── GCN Pre-trainer ──────────────────────────────

class GCNPretrainer:
    """
    Supervised pre-training of GCNEncoder + linear head on smell classification.
    Saves checkpoint; resumes if checkpoint exists.
    """

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.log    = logging.getLogger("SmellRL.pretrain")

        gcn_cfg = cfg["gcn"]
        self.gcn = GCNEncoder(
            in_ch  = gcn_cfg["node_feature_dim"],
            hidden = gcn_cfg["hidden_dim"],
            out_ch = gcn_cfg["output_dim"],
            dropout= gcn_cfg["dropout"],
        )
        self.classifier = GCNClassifier(self.gcn, n_classes=cfg["smells"]["n_classes"])
        self.classifier.to(device)

        pt_cfg         = cfg["gcn_pretrain"]
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=pt_cfg["learning_rate"])
        self.epochs    = pt_cfg["epochs"]
        self.batch_sz  = pt_cfg["batch_size"]
        self.ckpt_freq = pt_cfg.get("checkpoint_freq", 10)

        self.ckpt_mgr = CheckpointManager(cfg["paths"]["checkpoint_dir"], "gcn_pretrain", self.log)

    def train(self, train_ds: SmellDataset, val_ds: SmellDataset) -> GCNEncoder:
        """Run pre-training. Returns the trained GCN encoder."""
        start_epoch = 0
        history     = []
        state_info  = {}

        payload = self.ckpt_mgr.load_latest()
        if payload:
            start_epoch, state_info, history = self.ckpt_mgr.restore(
                payload,
                {"gcn": self.gcn, "classifier": self.classifier},
                {"optimizer": self.optimizer},
                self.device,
            )
            self.log.info(f"[Pretrain] Resumed from epoch {start_epoch}")

        self.log.info(f"[Pretrain] Training GCN encoder for {self.epochs} epochs")

        for epoch in range(start_epoch, self.epochs):
            t0         = time.time()
            train_loss = self._run_epoch(train_ds, train=True)
            val_loss, val_acc = self._run_epoch_eval(val_ds)
            elapsed    = time.time() - t0

            history.append({"epoch": epoch, "train_loss": train_loss,
                            "val_loss": val_loss, "val_acc": val_acc})

            self.log.info(
                f"[Pretrain] Epoch {epoch+1:3d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | {elapsed:.1f}s"
            )

            save_num = ((epoch + 1) % self.ckpt_freq == 0)
            self.ckpt_mgr.save(
                step      = epoch + 1,
                models    = {"gcn": self.gcn, "classifier": self.classifier},
                optimizers= {"optimizer": self.optimizer},
                state     = {"best_val_acc": state_info.get("best_val_acc", 0.0)},
                history   = history,
                save_numbered=save_num,
            )

        self.log.info("[Pretrain] GCN pre-training complete.")
        return self.gcn

    def _run_epoch(self, ds: SmellDataset, train: bool) -> float:
        self.classifier.train(train)
        total_loss = 0.0
        indices    = list(range(len(ds)))
        random.shuffle(indices)

        for start in range(0, len(indices), self.batch_sz):
            batch_idx = indices[start: start + self.batch_sz]
            batch_items = [ds[i] for i in batch_idx]

            self.optimizer.zero_grad()
            batch_loss = 0.0
            for item in batch_items:
                x          = item["x"].to(self.device)
                edge_index = item["edge_index"].to(self.device)
                y          = item["y"].unsqueeze(0).to(self.device)
                logits     = self.classifier(x, edge_index)
                loss       = F.cross_entropy(logits, y)
                batch_loss += loss

            batch_loss = batch_loss / len(batch_items)
            if train:
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
                self.optimizer.step()
            total_loss += batch_loss.item()

        return total_loss / max(1, len(indices) // self.batch_sz)

    def _run_epoch_eval(self, ds: SmellDataset) -> Tuple[float, float]:
        self.classifier.eval()
        total_loss = 0.0
        correct    = 0
        with torch.no_grad():
            for item in ds:
                x          = item["x"].to(self.device)
                edge_index = item["edge_index"].to(self.device)
                y          = item["y"].unsqueeze(0).to(self.device)
                logits     = self.classifier(x, edge_index)
                loss       = F.cross_entropy(logits, y)
                total_loss += loss.item()
                if logits.argmax(dim=1).item() == y.item():
                    correct += 1
        n = len(ds)
        return total_loss / max(n, 1), correct / max(n, 1)


# ──────────────────────────── Hierarchical DQN Trainer ─────────────────────

class HierarchicalTrainer:
    """
    Full RL training loop for SmellRL.

    Per episode (= one full pass through training set):
      1. For each instance: select action, compute reward, push to buffer
      2. After buffer warm-up: sample batch and update networks
      3. Sync target networks every target_update_freq steps
      4. Decay ε, log, checkpoint

    Resume: loads latest checkpoint at construction if it exists.
    """

    def __init__(self, agent: HierarchicalDQNAgent, cfg: dict, device: torch.device):
        self.agent   = agent
        self.cfg     = cfg
        self.device  = device
        self.log     = logging.getLogger("SmellRL.trainer")

        tcfg = cfg["training"]
        self.lr         = tcfg["learning_rate"]
        self.batch_sz   = tcfg["batch_size"]
        self.buf_cap    = tcfg["replay_buffer_size"]
        self.eps_start  = tcfg["epsilon_start"]
        self.eps_end    = tcfg["epsilon_end"]
        self.eps_decay  = tcfg["epsilon_decay_steps"]
        self.tgt_freq   = tcfg["target_update_freq"]
        self.max_eps    = tcfg["max_episodes"]
        self.warmup     = tcfg["warmup_steps"]
        self.ckpt_freq  = tcfg["checkpoint_freq"]
        self.log_freq   = tcfg["log_freq"]
        self.eval_freq  = tcfg["eval_freq"]

        self.reward_fn = RewardFunction(cfg["reward"])
        self.buffer    = ReplayBuffer(self.buf_cap)

        self.opt_q1 = torch.optim.Adam(
            list(agent.q1.parameters()) + list(agent.gcn.parameters()), lr=self.lr
        )
        self.opt_q2 = torch.optim.Adam(agent.q2.parameters(), lr=self.lr)

        self.ckpt_mgr  = CheckpointManager(cfg["paths"]["checkpoint_dir"], "smellrl", self.log)
        self.history: List[dict] = []
        self.global_step  = 0
        self.start_episode = 0
        self.epsilon       = self.eps_start
        self.best_val_f1   = 0.0

    def _compute_epsilon(self) -> float:
        frac = min(1.0, self.global_step / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - frac)

    def _try_resume(self):
        payload = self.ckpt_mgr.load_latest()
        if payload is None:
            return
        models = {
            "gcn": self.agent.gcn, "q1": self.agent.q1, "q2": self.agent.q2,
            "q1_target": self.agent.q1_target, "q2_target": self.agent.q2_target,
        }
        optimizers = {"opt_q1": self.opt_q1, "opt_q2": self.opt_q2}
        step, state, history = self.ckpt_mgr.restore(payload, models, optimizers, self.device)

        self.start_episode = step
        self.global_step   = state.get("global_step", 0)
        self.epsilon       = state.get("epsilon", self.eps_start)
        self.best_val_f1   = state.get("best_val_f1", 0.0)
        self.history       = history
        self.log.info(f"[Trainer] Resumed from episode={step}, global_step={self.global_step}, ε={self.epsilon:.4f}")

    def _save_checkpoint(self, episode: int, save_num: bool = False):
        models = {
            "gcn": self.agent.gcn, "q1": self.agent.q1, "q2": self.agent.q2,
            "q1_target": self.agent.q1_target, "q2_target": self.agent.q2_target,
        }
        optimizers = {"opt_q1": self.opt_q1, "opt_q2": self.opt_q2}
        state = {
            "global_step": self.global_step,
            "epsilon":     self.epsilon,
            "best_val_f1": self.best_val_f1,
        }
        self.ckpt_mgr.save(episode, models, optimizers, state, self.history, save_num)

    def train(
        self,
        train_ds: SmellDataset,
        val_ds:   SmellDataset,
    ) -> List[dict]:
        self._try_resume()

        self.log.info(
            f"[Trainer] Starting RL training: episodes={self.max_eps}, "
            f"start={self.start_episode}, warmup={self.warmup}"
        )

        train_indices = list(range(len(train_ds)))

        for episode in range(self.start_episode, self.max_eps):
            t0 = time.time()
            random.shuffle(train_indices)

            ep_reward   = 0.0
            ep_loss_q1  = 0.0
            ep_loss_q2  = 0.0
            p1_correct  = 0
            p2_correct  = 0
            n_updates   = 0

            self.log.debug(f"[Trainer] Episode {episode+1} start — ε={self.epsilon:.4f}")

            for idx in train_indices:
                item = train_ds[idx]
                true_smell = int(item["y"].item())

                # Select action
                a1, a2, state_vec = self.agent.select_action(item, self.epsilon)

                # Compute reward
                reward, components = self.reward_fn(true_smell, a1, a2)
                ep_reward += reward

                # Track accuracy
                if a1 == true_smell:
                    p1_correct += 1
                if a2 in SMELL_TO_PATTERNS.get(true_smell, []):
                    p2_correct += 1

                # Push transition
                self.buffer.push({
                    "state":  state_vec,
                    "a1":     a1,
                    "a2":     a2,
                    "reward": reward,
                })

                # Update networks after warm-up
                if len(self.buffer) >= self.warmup:
                    batch = self.buffer.sample(self.batch_sz)
                    lq1, lq2 = self.agent.train_step(batch, self.opt_q1, self.opt_q2)
                    ep_loss_q1 += lq1
                    ep_loss_q2 += lq2
                    n_updates  += 1

                # Target network sync
                if self.global_step > 0 and self.global_step % self.tgt_freq == 0:
                    self.agent.sync_targets()
                    self.log.debug(f"[Trainer] Targets synced at step {self.global_step}")

                self.epsilon = self._compute_epsilon()
                self.global_step += 1

            # ── Episode summary ───────────────────────────────────────────
            n = len(train_indices)
            ep_stats = {
                "episode":    episode + 1,
                "reward":     ep_reward / n,
                "p1_acc":     p1_correct / n,
                "p2_match":   p2_correct / n,
                "loss_q1":    ep_loss_q1 / max(1, n_updates),
                "loss_q2":    ep_loss_q2 / max(1, n_updates),
                "epsilon":    self.epsilon,
                "elapsed_s":  round(time.time() - t0, 1),
            }
            self.history.append(ep_stats)

            if (episode + 1) % self.log_freq == 0:
                self.log.info(
                    f"[Trainer] Ep {episode+1:4d}/{self.max_eps} | "
                    f"reward={ep_stats['reward']:+.4f} | "
                    f"p1_acc={ep_stats['p1_acc']:.4f} | "
                    f"p2_match={ep_stats['p2_match']:.4f} | "
                    f"loss_q1={ep_stats['loss_q1']:.4f} | "
                    f"loss_q2={ep_stats['loss_q2']:.4f} | "
                    f"ε={self.epsilon:.4f} | "
                    f"{ep_stats['elapsed_s']}s"
                )

            # ── Validation ────────────────────────────────────────────────
            if (episode + 1) % self.eval_freq == 0:
                val_f1 = self._evaluate(val_ds)
                self.log.info(f"[Trainer] Val F1 (smell) = {val_f1:.4f}")
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self._save_checkpoint(episode + 1, save_num=True)
                    self.log.info(f"[Trainer] New best val F1: {val_f1:.4f} — checkpoint saved")

            # ── Periodic checkpoint ───────────────────────────────────────
            if (episode + 1) % self.ckpt_freq == 0:
                self._save_checkpoint(episode + 1)

        self._save_checkpoint(self.max_eps)
        self.log.info("[Trainer] Training complete.")
        self._save_history()
        return self.history

    @torch.no_grad()
    def _evaluate(self, ds: SmellDataset) -> float:
        """Quick Phase-1 F1 on validation set."""
        from sklearn.metrics import f1_score
        self.agent.eval()
        y_true, y_pred = [], []
        for item in ds:
            a1, _, _ = self.agent.select_action(item, epsilon=0.0)
            y_true.append(int(item["y"].item()))
            y_pred.append(a1)
        self.agent.train()
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    def _save_history(self):
        results_dir = self.cfg["paths"]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        self.log.info(f"[Trainer] Training history saved to {path}")


# ──────────────────────────── Flat DQN Trainer ────────────────────────────

class FlatDQNTrainer:
    """Trains the flat DQN baseline (same protocol as HierarchicalTrainer)."""

    def __init__(self, agent: FlatDQNAgent, cfg: dict, device: torch.device):
        self.agent  = agent
        self.cfg    = cfg
        self.device = device
        self.log    = logging.getLogger("SmellRL.flat_trainer")

        tcfg = cfg["training"]
        self.lr        = tcfg["learning_rate"]
        self.batch_sz  = tcfg["batch_size"]
        self.eps_start = tcfg["epsilon_start"]
        self.eps_end   = tcfg["epsilon_end"]
        self.eps_decay = tcfg["epsilon_decay_steps"]
        self.tgt_freq  = tcfg["target_update_freq"]
        self.max_eps   = tcfg["max_episodes"]
        self.warmup    = tcfg["warmup_steps"]

        self.reward_fn = RewardFunction(cfg["reward"])
        self.buffer    = ReplayBuffer(tcfg["replay_buffer_size"])
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=self.lr)
        self.ckpt_mgr  = CheckpointManager(cfg["paths"]["checkpoint_dir"], "flat_dqn", self.log)
        self.history: List[dict] = []
        self.global_step = 0
        self.epsilon     = self.eps_start

    def _eps(self) -> float:
        frac = min(1.0, self.global_step / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - frac)

    def train(self, train_ds: SmellDataset) -> List[dict]:
        payload = self.ckpt_mgr.load_latest()
        start_ep = 0
        if payload:
            start_ep, state, hist = self.ckpt_mgr.restore(
                payload,
                {"agent": self.agent},
                {"optimizer": self.optimizer},
                self.device,
            )
            self.global_step = state.get("global_step", 0)
            self.epsilon     = state.get("epsilon", self.eps_start)
            self.history     = hist

        self.log.info(f"[FlatDQN] Training for {self.max_eps} episodes, start={start_ep}")
        indices = list(range(len(train_ds)))

        for episode in range(start_ep, self.max_eps):
            random.shuffle(indices)
            ep_reward = 0.0
            ep_loss   = 0.0
            p1c = p2c = n_upd = 0

            for idx in indices:
                item       = train_ds[idx]
                true_smell = int(item["y"].item())
                a1, a2     = self.agent.select_action(item, self.epsilon)
                reward, _  = self.reward_fn(true_smell, a1, a2)
                ep_reward += reward

                if a1 == true_smell: p1c += 1
                if a2 in SMELL_TO_PATTERNS.get(true_smell, []): p2c += 1

                self.buffer.push({
                    "x": item["x"], "edge_index": item["edge_index"],
                    "a1": a1, "a2": a2, "reward": reward,
                })

                if len(self.buffer) >= self.warmup:
                    batch = self.buffer.sample(self.batch_sz)
                    ep_loss  += self.agent.train_step(batch, self.optimizer)
                    n_upd    += 1

                if self.global_step > 0 and self.global_step % self.tgt_freq == 0:
                    self.agent.sync_target()

                self.epsilon = self._eps()
                self.global_step += 1

            n = len(indices)
            stats = {
                "episode":  episode + 1,
                "reward":   ep_reward / n,
                "p1_acc":   p1c / n,
                "p2_match": p2c / n,
                "loss":     ep_loss / max(1, n_upd),
                "epsilon":  self.epsilon,
            }
            self.history.append(stats)

            if (episode + 1) % self.cfg["training"]["log_freq"] == 0:
                self.log.info(
                    f"[FlatDQN] Ep {episode+1:4d} | "
                    f"reward={stats['reward']:+.4f} | "
                    f"p1_acc={stats['p1_acc']:.4f} | "
                    f"p2_match={stats['p2_match']:.4f}"
                )

            if (episode + 1) % self.cfg["training"]["checkpoint_freq"] == 0:
                self.ckpt_mgr.save(
                    episode + 1,
                    {"agent": self.agent},
                    {"optimizer": self.optimizer},
                    {"global_step": self.global_step, "epsilon": self.epsilon},
                    self.history,
                )

        self.log.info("[FlatDQN] Training complete.")
        return self.history
