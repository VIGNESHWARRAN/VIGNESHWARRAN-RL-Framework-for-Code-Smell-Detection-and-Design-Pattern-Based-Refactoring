import copy
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import logging

from src.utils import CheckpointManager
from src.models import GCNEncoder, RGATEncoder, SupervisedSmellDetector
from src.data import NUM_EDGE_TYPES

logger = logging.getLogger("SmellRL.training")


# ──────────────────────────── Focal Loss ───────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification on imbalanced datasets.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Key properties:
      - γ=0 reduces to standard cross-entropy (disable focusing).
      - γ=2 (default): down-weights easy examples by up to 100×,
        forcing the model's capacity toward hard, misclassified examples.
        This is exactly what the MLCQ dataset needs: NoSmell is trivially easy
        so standard CE wastes gradient budget on it; Focal Loss stops doing so.
      - α: per-class inverse-frequency weights, exactly as used in the old DQN
        reward scaling — now applied directly in the loss function instead.

    Args:
        gamma:     Focusing parameter (default 2.0).
        alpha:     Per-class weight tensor of shape [n_classes] or None.
        reduction: "mean" | "sum" | "none"
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits:  [B, n_classes]   (raw, un-normalised)
        # targets: [B]              (integer class indices)
        log_probs = F.log_softmax(logits, dim=-1)                          # [B, C]
        probs     = log_probs.exp()                                         # [B, C]

        log_pt    = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)   # [B]
        pt        = probs.gather(1, targets.unsqueeze(1)).squeeze(1)       # [B]

        focal_term = (1.0 - pt).pow(self.gamma)                            # [B]
        loss       = -focal_term * log_pt                                   # [B]

        if self.alpha is not None:
            alpha_t = self.alpha[targets]                                   # [B]
            loss    = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss   # "none"


# ──────────────────────── Replay Buffer (DQN ablation) ─────────────────────

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


# ─────────────────────── PRIMARY TRAINER: Supervised ──────────────────────

class SupervisedTrainer:
    """
    End-to-end supervised training of RGATEncoder + linear classifier
    using Focal Loss.

    This is the PRIMARY training pipeline, replacing the contextual bandit DQN
    formulation. The mathematical justification:
      - DQN (γ=0) is equivalent to a noisy, sample-inefficient classifier
        because the "reward" is just a hand-crafted proxy for cross-entropy.
      - Focal Loss is strictly more principled: it directly optimises the
        per-class log-probability weighted by a confidence-based focusing term,
        eliminating sample-efficiency overhead and reward engineering noise.

    DQNTrainer is retained below as an ablation baseline to explicitly prove
    why this transition was necessary in the paper.

    Features:
      - Per-class inverse-frequency α weighting in FocalLoss
      - AdamW optimiser with cosine annealing LR schedule
      - Gradient clipping (max norm 1.0)
      - Early stopping on validation accuracy
      - Best-model restoration after training
      - Checkpoint saving every `checkpoint_freq` epochs
    """
    def __init__(self, cfg: dict, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.log    = logging.getLogger("SmellRL.supervised")

        sup_cfg         = cfg.get("supervised_train", {})
        self.epochs     = sup_cfg.get("epochs", 150)
        self.batch_sz   = sup_cfg.get("batch_size", 32)
        self.lr         = sup_cfg.get("learning_rate", 3e-4)
        self.weight_decay = sup_cfg.get("weight_decay", 1e-5)
        self.ckpt_freq  = sup_cfg.get("checkpoint_freq", 25)
        self.patience   = sup_cfg.get("early_stopping_patience", 20)
        self.eval_freq  = sup_cfg.get("eval_freq", 10)
        self.focal_gamma = sup_cfg.get("focal_gamma",
                           cfg.get("gcn_pretrain", {}).get("focal_gamma", 2.0))

        self.n_classes = cfg["smells"]["n_classes"]
        self.ckpt_mgr  = CheckpointManager(
            cfg["paths"]["checkpoint_dir"], "supervised", self.log
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _build_encoder(self, feature_dim: int) -> RGATEncoder:
        gcn_cfg = self.cfg["gcn"]
        return RGATEncoder(
            in_ch=feature_dim,
            hidden=gcn_cfg["hidden_dim"],
            out_ch=gcn_cfg["output_dim"],
            num_edge_types=gcn_cfg.get("num_edge_types", NUM_EDGE_TYPES),
            num_heads=gcn_cfg.get("num_heads", 4),
            dropout=gcn_cfg["dropout"],
        ).to(self.device)

    def _compute_class_weights(self, train_ds) -> torch.Tensor:
        """Inverse-frequency weighting identical to old DQNTrainer reward scaling."""
        counts = {}
        for item in train_ds:
            y = int(item["y"].item())
            counts[y] = counts.get(y, 0) + 1
        total   = sum(counts.values())
        weights = torch.ones(self.n_classes)
        for cls_idx, count in counts.items():
            weights[cls_idx] = total / (self.n_classes * count) if count > 0 else 1.0
        self.log.info(f"[Supervised] Class weights for FocalLoss α: {weights.tolist()}")
        return weights.to(self.device)

    # ── Main training loop ───────────────────────────────────────────────────

    def train(
        self,
        train_ds,
        val_ds,
    ) -> Tuple[SupervisedSmellDetector, List[dict]]:
        """
        Train encoder + classifier end-to-end.

        Returns:
            detector: SupervisedSmellDetector loaded with best-val-acc weights.
            history:  List of per-epoch stat dicts.
        """
        feature_dim = train_ds[0]["x"].shape[1] if len(train_ds) > 0 else 783

        encoder    = self._build_encoder(feature_dim)
        classifier = nn.Linear(
            self.cfg["gcn"]["output_dim"], self.n_classes
        ).to(self.device)

        class_weights = self._compute_class_weights(train_ds)
        focal_loss    = FocalLoss(gamma=self.focal_gamma, alpha=class_weights)

        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(classifier.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        history          = []
        best_val_acc     = 0.0
        best_state       = None
        patience_counter = 0
        last_epoch       = 0

        self.log.info(
            f"[Supervised] Training for up to {self.epochs} epochs "
            f"(patience={self.patience}, eval_freq={self.eval_freq})"
        )
        self.log.info(
            f"[Supervised] Focal Loss γ={self.focal_gamma}, "
            f"LR={self.lr}, batch={self.batch_sz}"
        )

        for epoch in range(self.epochs):
            last_epoch = epoch
            encoder.train()
            classifier.train()

            indices = list(range(len(train_ds)))
            random.shuffle(indices)

            total_loss = 0.0
            correct    = 0
            batches    = 0

            for start in range(0, len(indices), self.batch_sz):
                batch_idx = indices[start: start + self.batch_sz]
                optimizer.zero_grad()

                batch_logits = []
                batch_labels = []

                for i in batch_idx:
                    item       = train_ds[i]
                    x          = item["x"].to(self.device)
                    edge_index = item["edge_index"].to(self.device)
                    edge_type  = item["edge_type"].to(self.device)
                    y          = item["y"].to(self.device)

                    emb    = encoder(x, edge_index, edge_type)
                    logits = classifier(emb)
                    batch_logits.append(logits)
                    batch_labels.append(y.unsqueeze(0))

                batch_logits = torch.cat(batch_logits, dim=0)   # [B, C]
                batch_labels = torch.cat(batch_labels, dim=0)   # [B]

                loss = focal_loss(batch_logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(classifier.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

                total_loss += loss.item()
                correct    += (batch_logits.argmax(dim=1) == batch_labels).sum().item()
                batches    += 1

            scheduler.step()

            train_loss = total_loss / max(1, batches)
            train_acc  = correct / max(1, len(train_ds))

            stats: dict = {
                "epoch":          epoch + 1,
                "train_loss":     train_loss,
                "train_accuracy": train_acc,
            }

            # ── Validation & early stopping ──────────────────────────────
            do_eval = (val_ds is not None) and ((epoch + 1) % self.eval_freq == 0)

            if do_eval:
                encoder.eval()
                classifier.eval()
                val_correct = 0
                val_loss    = 0.0
                with torch.no_grad():
                    for item in val_ds:
                        x          = item["x"].to(self.device)
                        edge_index = item["edge_index"].to(self.device)
                        edge_type  = item["edge_type"].to(self.device)
                        y          = item["y"].to(self.device)
                        emb        = encoder(x, edge_index, edge_type)
                        logits     = classifier(emb)
                        val_loss  += focal_loss(logits, y.unsqueeze(0)).item()
                        if logits.argmax(dim=1).item() == y.item():
                            val_correct += 1

                val_acc  = val_correct / max(1, len(val_ds))
                val_loss = val_loss    / max(1, len(val_ds))
                stats["val_accuracy"] = val_acc
                stats["val_loss"]     = val_loss

                self.log.info(
                    f"[Supervised] Ep {epoch+1:3d}/{self.epochs} | "
                    f"loss={train_loss:.4f} | acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
                )

                if val_acc > best_val_acc:
                    best_val_acc     = val_acc
                    best_state       = {
                        "encoder":    copy.deepcopy(encoder.state_dict()),
                        "classifier": copy.deepcopy(classifier.state_dict()),
                    }
                    patience_counter = 0
                    self.log.info(f"[Supervised] ✓ New best val_acc={val_acc:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        self.log.info(
                            f"[Supervised] Early stopping at epoch {epoch+1} "
                            f"(no improvement for {self.patience} eval cycles)"
                        )
                        history.append(stats)
                        break
            else:
                self.log.info(
                    f"[Supervised] Ep {epoch+1:3d}/{self.epochs} | "
                    f"loss={train_loss:.4f} | acc={train_acc:.4f}"
                )

            history.append(stats)

            # ── Periodic checkpoint ──────────────────────────────────────
            if (epoch + 1) % self.ckpt_freq == 0:
                self.ckpt_mgr.save(
                    step=epoch + 1,
                    models={"encoder": encoder, "classifier": classifier},
                    optimizers={"optimizer": optimizer},
                    state={"best_val_acc": best_val_acc},
                    history=history,
                    save_numbered=False,
                )

        # ── Restore best weights ─────────────────────────────────────────
        if best_state is not None:
            encoder.load_state_dict(best_state["encoder"])
            classifier.load_state_dict(best_state["classifier"])
            self.log.info(
                f"[Supervised] Restored best model (val_acc={best_val_acc:.4f})"
            )

        # Final checkpoint
        self.ckpt_mgr.save(
            step=last_epoch + 1,
            models={"encoder": encoder, "classifier": classifier},
            optimizers={"optimizer": optimizer},
            state={"best_val_acc": best_val_acc},
            history=history,
            save_numbered=False,
        )

        detector = SupervisedSmellDetector(
            encoder, classifier, self.n_classes, self.device
        )
        return detector, history


# ────────────────────── ABLATION: GCN Pre-trainer ──────────────────────────
# Kept for the ablation study section of the paper.
# SupervisedTrainer is the primary pipeline; GCNPretrainer is a sub-experiment.

class GCNPretrainer:
    """
    Supervised pre-training of the encoder on smell classification.

    Ablation baseline — demonstrates the effect of Focal Loss in isolation,
    independent of the RGAT architecture change.  Used by --stage pretrain.
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
        self.loss_type = pt_cfg.get("loss", "focal")
        self.focal_gamma = pt_cfg.get("focal_gamma", 2.0)
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
        encoder_type = gcn_cfg.get("encoder_type", "rgat")

        if encoder_type == "rgat":
            enc = RGATEncoder(
                in_ch=feature_dim,
                hidden=gcn_cfg["hidden_dim"],
                out_ch=gcn_cfg["output_dim"],
                num_edge_types=gcn_cfg.get("num_edge_types", NUM_EDGE_TYPES),
                num_heads=gcn_cfg.get("num_heads", 4),
                dropout=gcn_cfg["dropout"],
            ).to(self.device)
        else:
            enc = GCNEncoder(
                in_ch=feature_dim,
                hidden=gcn_cfg["hidden_dim"],
                out_ch=gcn_cfg["output_dim"],
                dropout=gcn_cfg["dropout"],
            ).to(self.device)

        n_classes  = self.cfg["smells"]["n_classes"]
        classifier = nn.Linear(gcn_cfg["output_dim"], n_classes).to(self.device)
        optimizer  = torch.optim.Adam(
            list(enc.parameters()) + list(classifier.parameters()), lr=self.lr
        )

        # Loss function
        if self.loss_type == "focal":
            class_weights = self._compute_class_weights(train_ds)
            criterion = FocalLoss(gamma=self.focal_gamma, alpha=class_weights)
            self.log.info(
                f"[Pretrain] Using FocalLoss (γ={self.focal_gamma}) "
                f"with class weights: {class_weights.cpu().tolist()}"
            )
        else:
            criterion = nn.CrossEntropyLoss()
            self.log.info("[Pretrain] Using CrossEntropyLoss")

        self.log.info(
            f"[Pretrain] Training {encoder_type.upper()} encoder "
            f"for {self.epochs} epochs"
        )

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
                    y          = item["y"].unsqueeze(0).to(self.device)

                    if encoder_type == "rgat":
                        edge_type = item["edge_type"].to(self.device)
                        emb = enc(x, edge_index, edge_type)
                    else:
                        emb = enc(x, edge_index)

                    logits     = classifier(emb)
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
                    y          = item["y"].unsqueeze(0).to(self.device)

                    if encoder_type == "rgat":
                        edge_type = item["edge_type"].to(self.device)
                        emb = enc(x, edge_index, edge_type)
                    else:
                        emb = enc(x, edge_index)

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


# ────────────────────── ABLATION: DQN Trainer ──────────────────────────────
# Retained to reproduce contextual-bandit baseline numbers in the paper.
# NOT used in the primary --stage train pipeline.

class DQNTrainer:
    """
    Contextual-bandit DQN trainer (γ=0).

    Ablation baseline: demonstrates why this formulation caps at ~50% accuracy
    on MLCQ compared to SupervisedTrainer with Focal Loss.

    Architecture differences from primary pipeline:
      - Loss:     Huber on Q-values  vs.  Focal Loss on logits
      - Gradient: per-step stochastic  vs.  batched mini-batch
      - Sample:   replay buffer  vs.  full epoch
    """
    def __init__(self, agent, cfg: dict, device: torch.device):
        self.agent  = agent
        self.cfg    = cfg
        self.device = device

        r_cfg = cfg["reward"]
        self.r_correct_smell   = r_cfg["correct_smell"]
        self.r_correct_nosmell = r_cfg["correct_nosmell"]
        self.r_incorrect_smell = r_cfg["incorrect_smell"]
        self.r_false_alarm     = r_cfg["false_alarm"]
        self.r_missed_smell    = r_cfg["missed_smell"]

        self.nosmell_idx = cfg["smells"]["classes"].index("NoSmell")

        tcfg            = cfg["dqn"]
        self.batch_sz   = tcfg["batch_size"]
        self.warmup     = tcfg["warmup_steps"]
        self.tgt_freq   = tcfg["target_update_freq"]
        self.eps_start  = tcfg["epsilon_start"]
        self.eps_end    = tcfg["epsilon_end"]
        self.eps_decay  = tcfg["epsilon_decay_steps"]
        self.max_eps    = tcfg["max_episodes"]

        self.eval_freq  = cfg.get("training", {}).get("eval_freq", 10)
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
            cfg["paths"]["checkpoint_dir"], "smellrl_dqn", logger
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
        logger.info(f"[DQN-Ablation] Class weights for reward scaling: {weights}")
        return weights

    def _eps(self) -> float:
        frac = min(1.0, self.global_step / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - frac)

    def _compute_reward(self, action: int, true_smell: int) -> float:
        # Rows represent Ground-Truth Smells: [GodClass, FeatureEnvy, LongMethod, DataClass, NoSmell]
        # Cols represent Refactoring Action Patterns: [Strategy, Observer, Facade, Mediator, ExtractClass, None]
        # Soft Reward Matrix mapping code quality feedback to refactoring decisions:
        soft_reward_matrix = [
            [-1.0, -1.0,  2.0,  1.0,  0.5, -2.0],  # GodClass
            [ 2.0, -1.0, -1.0, -1.0,  1.0, -2.0],  # FeatureEnvy
            [-1.0, -1.0, -1.0, -1.0,  2.0, -2.0],  # LongMethod
            [-1.0,  2.0, -1.0, -1.0, -1.0, -2.0],  # DataClass
            [-1.0, -1.0, -1.0, -1.0, -1.0,  2.0],  # NoSmell
        ]
        
        if true_smell < 0 or true_smell >= len(soft_reward_matrix):
            base = -2.0
        else:
            pattern_idx = action
            if pattern_idx < 0 or pattern_idx >= len(soft_reward_matrix[0]):
                base = -2.0
            else:
                base = soft_reward_matrix[true_smell][pattern_idx]

        return base * self.class_weights.get(true_smell, 1.0)

    def train_step(self, batch: List[dict]) -> float:
        states  = torch.cat([b["state"] for b in batch], dim=0)
        actions = torch.tensor([b["action"] for b in batch], device=self.device)
        rewards = torch.tensor(
            [b["reward"] for b in batch], device=self.device, dtype=torch.float32
        )
        q_pred = self.agent.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss   = F.smooth_l1_loss(q_pred, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train(self, train_ds, val_ds=None) -> List[dict]:
        self.class_weights = self._compute_class_weights(train_ds)
        indices            = list(range(len(train_ds)))
        history            = []

        logger.info(
            f"[DQN-Ablation] Starting RL training for {self.max_eps} episodes..."
        )

        for episode in range(self.max_eps):
            random.shuffle(indices)
            ep_reward, ep_loss, correct = 0.0, 0.0, 0
            n_updates = 0

            for idx in indices:
                item       = train_ds[idx]
                true_smell = int(item["y"].item())

                action, state_vec = self.agent.select_action(item, self.epsilon)
                reward             = self._compute_reward(action, true_smell)

                ep_reward += reward
                # A refactoring action is considered "correct" if it is valid (reward > 0.0)
                valid_patterns = {
                    0: [2, 3, 4],  # GodClass -> Facade, Mediator, ExtractClass
                    1: [0, 4],     # FeatureEnvy -> Strategy, ExtractClass
                    2: [4],        # LongMethod -> ExtractClass
                    3: [1],        # DataClass -> Observer
                    4: [5]         # NoSmell -> None
                }
                if action in valid_patterns.get(true_smell, []):
                    correct += 1

                self.buffer.push(
                    {"state": state_vec, "action": action, "reward": reward}
                )

                if len(self.buffer) >= self.warmup:
                    batch      = self.buffer.sample(self.batch_sz)
                    ep_loss   += self.train_step(batch)
                    n_updates += 1

                if self.global_step > 0 and self.global_step % self.tgt_freq == 0:
                    self.agent.sync_target()

                self.epsilon     = self._eps()
                self.global_step += 1

            self.scheduler.step()

            stats = {
                "episode":  episode + 1,
                "reward":   ep_reward / len(indices),
                "accuracy": correct   / len(indices),
                "loss":     ep_loss   / max(1, n_updates),
                "epsilon":  self.epsilon,
            }

            if val_ds is not None and (episode + 1) % self.eval_freq == 0:
                self.agent.eval()
                val_correct = 0
                with torch.no_grad():
                    for val_item in val_ds:
                        val_action, _ = self.agent.select_action(val_item, epsilon=0.0)
                        valid_patterns = {
                            0: [2, 3, 4],  # GodClass -> Facade, Mediator, ExtractClass
                            1: [0, 4],     # FeatureEnvy -> Strategy, ExtractClass
                            2: [4],        # LongMethod -> ExtractClass
                            3: [1],        # DataClass -> Observer
                            4: [5]         # NoSmell -> None
                        }
                        if val_action in valid_patterns.get(int(val_item["y"].item()), []):
                            val_correct += 1
                val_acc              = val_correct / max(1, len(val_ds))
                stats["val_accuracy"] = val_acc
                self.agent.train()
                logger.info(f"[DQN-Ablation] Ep {episode+1:4d} | Val ACC: {val_acc:.4f}")

            history.append(stats)

            if (episode + 1) % self.cfg["training"]["log_freq"] == 0:
                logger.info(
                    f"[DQN-Ablation] Ep {episode+1:4d} | "
                    f"RWD: {stats['reward']:+.4f} | ACC: {stats['accuracy']:.4f} | "
                    f"LOSS: {stats['loss']:.4f} | EPS: {self.epsilon:.4f}"
                )

            if (episode + 1) % self.cfg["training"]["checkpoint_freq"] == 0 or \
               (episode + 1) == self.max_eps:
                self.ckpt_mgr.save(
                    step=episode + 1,
                    models={"agent": self.agent},
                    optimizers={"optimizer": self.optimizer},
                    state={"epsilon": self.epsilon},
                    history=history,
                    save_numbered=False,
                )

        return history