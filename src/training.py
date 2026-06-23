import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
import logging

from src.utils import CheckpointManager
from src.models import GCNEncoder

logger = logging.getLogger("SmellRL.training")

# ──────────────────────────── Replay Buffer ────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = []
        self.capacity = capacity
        self.idx = 0

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


# ──────────────────────────── GCN Pre-trainer ──────────────────────────────

class GCNPretrainer:
    """Supervised pre-training of the GCN encoder on smell classification."""
    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.log = logging.getLogger("SmellRL.pretrain")

        pt_cfg = cfg["gcn_pretrain"]
        self.epochs = pt_cfg["epochs"]
        self.batch_sz = pt_cfg["batch_size"]
        self.lr = pt_cfg["learning_rate"]
        self.ckpt_freq = pt_cfg.get("checkpoint_freq", 10)
        self.ckpt_mgr = CheckpointManager(cfg["paths"]["checkpoint_dir"], "gcn_pretrain", self.log)

    def train(self, train_ds, val_ds) -> GCNEncoder:
        # Dynamically detect feature dimension
        feature_dim = train_ds[0]["x"].shape[1] if len(train_ds) > 0 else self.cfg["ck_metrics"]["n_features"]
        
        gcn_cfg = self.cfg["gcn"]
        gcn = GCNEncoder(
            in_ch=feature_dim,
            hidden=gcn_cfg["hidden_dim"],
            out_ch=gcn_cfg["output_dim"],
            dropout=gcn_cfg["dropout"]
        ).to(self.device)
        
        n_classes = self.cfg["smells"]["n_classes"]
        classifier = nn.Linear(gcn_cfg["output_dim"], n_classes).to(self.device)
        
        optimizer = torch.optim.Adam(list(gcn.parameters()) + list(classifier.parameters()), lr=self.lr)
        
        self.log.info(f"[Pretrain] Training GCN encoder for {self.epochs} epochs")
        
        for epoch in range(self.epochs):
            # Training Phase
            gcn.train()
            classifier.train()
            total_loss = 0.0
            indices = list(range(len(train_ds)))
            random.shuffle(indices)
            
            for start in range(0, len(indices), self.batch_sz):
                batch_idx = indices[start: start + self.batch_sz]
                batch_loss = 0.0
                
                optimizer.zero_grad()
                for i in batch_idx:
                    item = train_ds[i]
                    x = item["x"].to(self.device)
                    edge_index = item["edge_index"].to(self.device)
                    y = item["y"].unsqueeze(0).to(self.device)
                    
                    emb = gcn(x, edge_index)
                    logits = classifier(emb)
                    loss = F.cross_entropy(logits, y)
                    batch_loss += loss
                    
                batch_loss = batch_loss / len(batch_idx)
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                
            train_loss = total_loss / max(1, len(indices) // self.batch_sz)
            
            # Validation Phase
            gcn.eval()
            classifier.eval()
            val_loss, correct = 0.0, 0
            with torch.no_grad():
                for item in val_ds:
                    x = item["x"].to(self.device)
                    edge_index = item["edge_index"].to(self.device)
                    y = item["y"].unsqueeze(0).to(self.device)
                    
                    emb = gcn(x, edge_index)
                    logits = classifier(emb)
                    val_loss += F.cross_entropy(logits, y).item()
                    if logits.argmax(dim=1).item() == y.item():
                        correct += 1
                        
            val_loss /= max(1, len(val_ds))
            val_acc = correct / max(1, len(val_ds))
            
            self.log.info(f"[Pretrain] Epoch {epoch+1:2d}/{self.epochs} | loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
            
            # Save Checkpoint
            if (epoch + 1) % self.ckpt_freq == 0 or (epoch + 1) == self.epochs:
                self.ckpt_mgr.save(
                    step=epoch + 1,
                    models={"gcn": gcn, "classifier": classifier},
                    optimizers={"optimizer": optimizer},
                    state={"val_acc": val_acc},
                    history=[],
                    save_numbered=False
                )
                
        return gcn


# ──────────────────────────── DQN Trainer ──────────────────────────────────

class DQNTrainer:
    def __init__(self, agent, cfg: dict, device: torch.device):
        self.agent = agent
        self.cfg = cfg
        self.device = device
        self.r_correct = cfg["reward"]["correct_smell"]
        self.r_incorrect = cfg["reward"]["incorrect_smell"]
        
        tcfg = cfg["dqn"]
        self.batch_sz = tcfg["batch_size"]
        self.warmup = tcfg["warmup_steps"]
        self.tgt_freq = tcfg["target_update_freq"]
        self.eps_start = tcfg["epsilon_start"]
        self.eps_end = tcfg["epsilon_end"]
        self.eps_decay = tcfg["epsilon_decay_steps"]
        self.max_eps = tcfg["max_episodes"]
        
        self.buffer = ReplayBuffer(tcfg["replay_buffer_size"])
        self.optimizer = torch.optim.Adam(
            list(agent.q.parameters()) + list(agent.gcn.parameters()), 
            lr=tcfg["learning_rate"]
        )
        self.global_step = 0
        self.epsilon = self.eps_start
        self.ckpt_mgr = CheckpointManager(cfg["paths"]["checkpoint_dir"], "smellrl", logger)

    def _eps(self) -> float:
        frac = min(1.0, self.global_step / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - frac)

    def train_step(self, batch: List[dict]) -> float:
        states = torch.cat([b["state"] for b in batch], dim=0)
        actions = torch.tensor([b["action"] for b in batch], device=self.device)
        rewards = torch.tensor([b["reward"] for b in batch], device=self.device, dtype=torch.float32)
        
        # Single-step episode: Target Q is exactly the reward.
        q_pred = self.agent.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_pred, rewards)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train(self, train_ds) -> List[dict]:
        indices = list(range(len(train_ds)))
        history = []
        
        logger.info(f"[Trainer] Starting RL training for {self.max_eps} episodes...")
        
        for episode in range(self.max_eps):
            random.shuffle(indices)
            ep_reward, ep_loss, correct = 0.0, 0.0, 0
            n_updates = 0
            
            for idx in indices:
                item = train_ds[idx]
                true_smell = int(item["y"].item())
                
                action, state_vec = self.agent.select_action(item, self.epsilon)
                reward = self.r_correct if action == true_smell else self.r_incorrect
                
                ep_reward += reward
                if action == true_smell: correct += 1
                
                self.buffer.push({"state": state_vec, "action": action, "reward": reward})
                
                if len(self.buffer) >= self.warmup:
                    batch = self.buffer.sample(self.batch_sz)
                    ep_loss += self.train_step(batch)
                    n_updates += 1
                
                if self.global_step > 0 and self.global_step % self.tgt_freq == 0:
                    self.agent.sync_target()
                    
                self.epsilon = self._eps()
                self.global_step += 1
                
            stats = {
                "episode": episode + 1,
                "reward": ep_reward / len(indices),
                "accuracy": correct / len(indices),
                "loss": ep_loss / max(1, n_updates),
                "epsilon": self.epsilon
            }
            history.append(stats)
            
            if (episode + 1) % self.cfg["training"]["log_freq"] == 0:
                logger.info(f"[Trainer] Ep {episode+1:4d} | RWD: {stats['reward']:+.4f} | ACC: {stats['accuracy']:.4f} | LOSS: {stats['loss']:.4f} | EPS: {self.epsilon:.4f}")
                
            if (episode + 1) % self.cfg["training"]["checkpoint_freq"] == 0 or (episode + 1) == self.max_eps:
                self.ckpt_mgr.save(
                    step=episode + 1,
                    models={"agent": self.agent},
                    optimizers={"optimizer": self.optimizer},
                    state={"epsilon": self.epsilon},
                    history=history,
                    save_numbered=False
                )
                
        return history