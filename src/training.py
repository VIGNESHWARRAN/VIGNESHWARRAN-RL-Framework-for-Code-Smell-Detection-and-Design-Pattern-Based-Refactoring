import copy
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import logging

from src.utils import CheckpointManager
from src.models import RGATEncoder, SmellDetectionAgent
from src.environment import apply_simulated_refactoring, quality_delta_reward, SOFT_REWARD_MATRIX

logger = logging.getLogger("SmellRL.training")

# ──────────────────────── Replay Buffer (DQN) ─────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer   = []
        self.capacity = capacity
        self.idx      = 0

    def push(self, transition: dict):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> List[dict]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ─────────────────────── GCN Pre-trainer ──────────────────────────

class GCNPretrainer:
    """
    Supervised pre-training of the RGAT encoder on smell classification.
    Warm-starts the encoder weights before RL begins.
    """
    def __init__(self, cfg: dict, device: torch.device):
        self.cfg      = cfg
        self.device   = device
        self.log      = logging.getLogger("SmellRL.pretrain")

        pt_cfg        = cfg["gcn_pretrain"]
        self.epochs   = pt_cfg["epochs"]
        self.batch_sz = pt_cfg["batch_size"]
        self.lr       = pt_cfg["learning_rate"]
        self.ckpt_freq = pt_cfg.get("checkpoint_freq", 10)
        self.ckpt_mgr = CheckpointManager(
            cfg["paths"]["checkpoint_dir"], "gcn_pretrain", self.log
        )

    def _compute_class_weights(self, train_ds) -> torch.Tensor:
        counts = {}
        for item in train_ds:
            y = int(item["y"].item())
            counts[y] = counts.get(y, 0) + 1
        n_classes = self.cfg["smells"]["n_classes"]
        total     = sum(counts.values())
        weights   = torch.ones(n_classes)
        for cls_idx, count in counts.items():
            weights[cls_idx] = total / (n_classes * count) if count > 0 else 1.0
        return weights.to(self.device)

    def train(self, train_ds, val_ds) -> RGATEncoder:
        feature_dim = train_ds[0]["x"].shape[1] if len(train_ds) > 0 else 783
        gcn_cfg     = self.cfg["gcn"]

        enc = RGATEncoder(
            in_ch=feature_dim,
            hidden=gcn_cfg["hidden_dim"],
            out_ch=gcn_cfg["output_dim"],
            num_edge_types=gcn_cfg.get("num_edge_types", 3),
            num_heads=gcn_cfg.get("num_heads", 4),
            dropout=gcn_cfg["dropout"],
        ).to(self.device)

        n_classes  = self.cfg["smells"]["n_classes"]
        classifier = nn.Linear(gcn_cfg["output_dim"], n_classes).to(self.device)
        optimizer  = torch.optim.Adam(
            list(enc.parameters()) + list(classifier.parameters()), lr=self.lr
        )

        class_weights = self._compute_class_weights(train_ds)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.log.info(f"[Pretrain] Using CrossEntropyLoss with weights: {class_weights.cpu().tolist()}")

        self.log.info(f"[Pretrain] Training RGAT encoder for {self.epochs} epochs")

        for epoch in range(self.epochs):
            enc.train()
            classifier.train()
            total_loss = 0.0
            indices    = list(range(len(train_ds)))
            random.shuffle(indices)

            for start in range(0, len(indices), self.batch_sz):
                batch_idx  = indices[start: start + self.batch_sz]
                batch_loss = torch.tensor(0.0, device=self.device)
                optimizer.zero_grad()

                for i in batch_idx:
                    item       = train_ds[i]
                    x          = item["x"].to(self.device)
                    edge_index = item["edge_index"].to(self.device)
                    edge_type  = item["edge_type"].to(self.device)
                    y          = item["y"].unsqueeze(0).to(self.device)

                    emb = enc(x, edge_index, edge_type)
                    logits = classifier(emb)
                    batch_loss = batch_loss + criterion(logits, y)

                batch_loss = batch_loss / len(batch_idx)
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()

            train_loss = total_loss / max(1, len(indices) // self.batch_sz)

            # Validation
            enc.eval()
            classifier.eval()
            val_loss, correct = 0.0, 0
            with torch.no_grad():
                for item in val_ds:
                    x          = item["x"].to(self.device)
                    edge_index = item["edge_index"].to(self.device)
                    edge_type  = item["edge_type"].to(self.device)
                    y          = item["y"].unsqueeze(0).to(self.device)

                    emb = enc(x, edge_index, edge_type)
                    logits    = classifier(emb)
                    val_loss += criterion(logits, y).item()
                    if logits.argmax(dim=1).item() == y.item():
                        correct += 1

            val_loss /= max(1, len(val_ds))
            val_acc   = correct / max(1, len(val_ds))

            self.log.info(
                f"[Pretrain] Epoch {epoch+1:2d}/{self.epochs} | "
                f"loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

            if (epoch + 1) % self.ckpt_freq == 0 or (epoch + 1) == self.epochs:
                self.ckpt_mgr.save(
                    step=epoch + 1,
                    models={"gcn": enc, "classifier": classifier},
                    optimizers={"optimizer": optimizer},
                    state={"val_acc": val_acc},
                    history=[],
                    save_numbered=False,
                )

        return enc


# ────────────────────────── DQNTrainer (PRIMARY) ────────────────────────────

class DQNTrainer:
    """
    DQN-based refactoring recommendation agent.
    Primary RL Trainer. Implements a genuine 2-step MDP with gamma=0.9.
    """
    def __init__(self, agent: SmellDetectionAgent, cfg: dict, device: torch.device):
        self.agent  = agent
        self.cfg    = cfg
        self.device = device
        self.log    = logging.getLogger("SmellRL.training")

        tcfg            = cfg["dqn"]
        self.gamma      = tcfg.get("gamma", 0.9)
        if self.gamma == 0.0:
            raise ValueError(
                "gamma=0.0 detected in config. This reduces the MDP to a contextual bandit "
                "(mathematically equivalent to weighted cross-entropy). Set gamma >= 0.9 "
                "for a genuine multi-step RL formulation. See implementation_plan.md."
            )
        
        self.log.info(f"[DQN] γ={self.gamma} — 2-step MDP enabled (genuine RL)")
        self.log.info(f"[DQN] Soft reward matrix loaded.")

        self.batch_sz   = tcfg["batch_size"]
        self.warmup     = tcfg["warmup_steps"]
        self.tgt_freq   = tcfg["target_update_freq"]
        self.eps_start  = tcfg["epsilon_start"]
        self.eps_end    = tcfg["epsilon_end"]
        self.eps_decay  = tcfg["epsilon_decay_steps"]
        self.max_eps    = tcfg["max_episodes"]
        self.eval_freq  = tcfg.get("eval_freq", 25)

        self.buffer     = ReplayBuffer(tcfg["replay_buffer_size"])

        weight_decay = tcfg.get("weight_decay", 1e-5)
        self.optimizer = torch.optim.Adam(
            list(agent.q.parameters()) + list(agent.gcn.parameters()),
            lr=tcfg["learning_rate"],
            weight_decay=weight_decay,
        )
        self.scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_eps
        )
        self.global_step  = 0
        self.epsilon      = self.eps_start
        self.ckpt_mgr     = CheckpointManager(
            cfg["paths"]["checkpoint_dir"], "smellrl_dqn", self.log
        )
        self.class_weights: dict = {}

    def _compute_class_weights(self, train_ds) -> dict:
        counts = {}
        for item in train_ds:
            y = int(item["y"].item())
            counts[y] = counts.get(y, 0) + 1
        n_classes = self.cfg["smells"]["n_classes"]
        total     = sum(counts.values())
        weights   = {
            cls_idx: total / (n_classes * count) if count > 0 else 1.0
            for cls_idx, count in counts.items()
        }
        self.log.info(f"[DQN] Inverse-frequency class weights computed: {weights}")
        return weights

    def _eps(self) -> float:
        frac = min(1.0, self.global_step / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - frac)

    def _compute_reward(self, action: int, true_smell: int) -> float:
        if true_smell < 0 or true_smell >= len(SOFT_REWARD_MATRIX):
            base = -2.0
        else:
            if action < 0 or action >= 6:
                base = -2.0
            else:
                base = SOFT_REWARD_MATRIX[true_smell][action]
        return base * self.class_weights.get(true_smell, 1.0)

    def train_step(self, batch: List[dict]) -> float:
        """Runs standard DQN loss update (Huber loss)."""
        states      = torch.cat([b["state"] for b in batch], dim=0)
        actions     = torch.tensor([b["action"] for b in batch], device=self.device)
        rewards     = torch.tensor([b["reward"] for b in batch], device=self.device, dtype=torch.float32)
        next_states = torch.cat([b["next_state"] for b in batch], dim=0)
        dones       = torch.tensor([b["done"] for b in batch], device=self.device, dtype=torch.float32)

        # Q(s, a)
        q_pred = self.agent.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target = r + gamma * max Q_target(s_next) * (1 - done)
        with torch.no_grad():
            q_next = self.agent.q_target(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * q_next * (1.0 - dones)

        loss = F.smooth_l1_loss(q_pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train(self, train_ds, val_ds=None) -> List[dict]:
        self.class_weights = self._compute_class_weights(train_ds)
        indices            = list(range(len(train_ds)))
        history            = []

        self.log.info(f"[DQN] Starting RL training for {self.max_eps} episodes...")

        for episode in range(self.max_eps):
            random.shuffle(indices)
            ep_reward, ep_loss, correct = 0.0, 0.0, 0
            n_updates = 0
            total_samples = len(indices)

            for step_idx, idx in enumerate(indices):
                item = train_ds[idx]
                true_smell = int(item["y"].item())

                # --- STEP 1: Observe original code, pick refactoring ---
                # select_action returns (action, state_vec)
                a1, s1_vec = self.agent.select_action(item, self.epsilon)
                r1 = self._compute_reward(a1, true_smell)
                ep_reward += r1

                valid_patterns = {
                    0: [2, 3, 4],  # GodClass -> Facade, Mediator, ExtractClass
                    1: [0, 4],     # FeatureEnvy -> Strategy, ExtractClass
                    2: [4],        # LongMethod -> ExtractClass
                    3: [1],        # DataClass -> Observer
                    4: [5]         # NoSmell -> None
                }
                if a1 in valid_patterns.get(true_smell, []):
                    correct += 1

                # --- STEP 2: Transition - apply simulated refactoring ---
                item2 = apply_simulated_refactoring(item, a1)

                # Observe state2 and pick refinement action
                a2, s2_vec = self.agent.select_action(item2, self.epsilon * 0.5)
                r2 = quality_delta_reward(item, item2)

                # Push Step 1 transition: s1 -> a1 -> r1 -> s2 (done=False)
                self.buffer.push({
                    "state": s1_vec,
                    "action": a1,
                    "reward": r1,
                    "next_state": s2_vec,
                    "done": False
                })

                # Push Step 2 transition: s2 -> a2 -> r2 -> s2 (done=True)
                self.buffer.push({
                    "state": s2_vec,
                    "action": a2,
                    "reward": r2,
                    "next_state": s2_vec,
                    "done": True
                })

                # Training step
                if len(self.buffer) >= self.warmup:
                    batch      = self.buffer.sample(self.batch_sz)
                    ep_loss   += self.train_step(batch)
                    n_updates += 1

                if self.global_step > 0 and self.global_step % self.tgt_freq == 0:
                    self.agent.sync_target()

                self.epsilon     = self._eps()
                self.global_step += 1

            self.scheduler.step()

            avg_reward = ep_reward / total_samples
            avg_acc    = correct / total_samples
            avg_loss   = ep_loss / max(1, n_updates)

            stats = {
                "episode":  episode + 1,
                "reward":   avg_reward,
                "accuracy": avg_acc,
                "loss":     avg_loss,
                "epsilon":  self.epsilon,
            }

            self.log.info(
                f"[DQN][Ep {episode+1}/{self.max_eps}] RWD: {avg_reward:+.4f} | "
                f"ACC: {avg_acc:.4f} | LOSS: {avg_loss:.5f} | EPS: {self.epsilon:.4f}"
            )

            # Validation
            if val_ds is not None and (episode + 1) % self.eval_freq == 0:
                self.agent.eval()
                val_correct = 0
                with torch.no_grad():
                    for val_item in val_ds:
                        val_action, _ = self.agent.select_action(val_item, epsilon=0.0)
                        valid_patterns = {
                            0: [2, 3, 4],
                            1: [0, 4],
                            2: [4],
                            3: [1],
                            4: [5]
                        }
                        if val_action in valid_patterns.get(int(val_item["y"].item()), []):
                            val_correct += 1
                val_acc = val_correct / max(1, len(val_ds))
                stats["val_accuracy"] = val_acc
                self.agent.train()
                self.log.info(f"[DQN][Ep {episode+1}] Validation Accuracy: {val_acc:.4f}")

            history.append(stats)

            if (episode + 1) % self.cfg["training"]["checkpoint_freq"] == 0 or (episode + 1) == self.max_eps:
                self.ckpt_mgr.save(
                    step=episode + 1,
                    models={"agent": self.agent},
                    optimizers={"optimizer": self.optimizer},
                    state={"epsilon": self.epsilon},
                    history=history,
                    save_numbered=False,
                )

        return history