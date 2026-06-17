"""
Neural network models for SmellRL.

GCNEncoder     — 2-layer GCN, produces 128-dim state vector per instance
UpperQNetwork  — Q₁(s)       → Q-values for 5 smell actions   (128→64→32→5)
LowerQNetwork  — Q₂(s,a₁)   → Q-values for 6 pattern actions (133→64→32→6)
FlatQNetwork   — flat DQN baseline                             (128→64→32→30)
HierarchicalDQNAgent — wraps Upper+Lower with target networks and action selection
"""

import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("SmellRL.models")

# ──────────────────────────── GCN Encoder ──────────────────────────────────

class GCNLayer(nn.Module):
    """Single GCN layer: A_hat @ X @ W with ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x          : [N, in_ch]
        edge_index : [2, E]
        """
        n = x.size(0)
        if edge_index.numel() == 0:
            return F.relu(self.linear(x))

        row, col = edge_index[0], edge_index[1]

        # degree-normalised adjacency (symmetric normalisation)
        deg = torch.zeros(n, device=x.device).scatter_add(0, row, torch.ones(row.size(0), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e6)
        deg_inv_sqrt[deg == 0] = 0.0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), x[row] * norm.unsqueeze(1))
        # add self-loops
        agg = agg + x

        return F.relu(self.linear(agg))


class GCNEncoder(nn.Module):
    """
    2-layer GCN → global mean pool → 128-dim state vector.
    Used as the shared feature extractor for both Q-networks.
    """

    def __init__(self, in_ch: int = 9, hidden: int = 64, out_ch: int = 128, dropout: float = 0.1):
        super().__init__()
        self.conv1   = GCNLayer(in_ch, hidden)
        self.conv2   = GCNLayer(hidden, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.out_ch  = out_ch
        logger.debug(f"[GCNEncoder] in={in_ch} hidden={hidden} out={out_ch}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns [1, out_ch] embedding for a single graph."""
        h = self.conv1(x, edge_index)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        # global mean pool over all nodes
        emb = h.mean(dim=0, keepdim=True)   # [1, out_ch]
        return emb


# ──────────────────────────── MLP helper ───────────────────────────────────

def _mlp(dims: List[int], dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ──────────────────────────── Q-Networks ───────────────────────────────────

class UpperQNetwork(nn.Module):
    """Q₁(s) → Q-values for 5 smell detection actions."""

    def __init__(self, state_dim: int = 128, hidden: List[int] = None, n_actions: int = 5):
        super().__init__()
        hidden = hidden or [64, 32]
        self.net = _mlp([state_dim] + hidden + [n_actions])
        logger.debug(f"[UpperQNet] dims={[state_dim]+hidden+[n_actions]}")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class LowerQNetwork(nn.Module):
    """Q₂(s, a₁_onehot) → Q-values for 6 pattern actions."""

    def __init__(self, state_dim: int = 128, n_smell_actions: int = 5,
                 hidden: List[int] = None, n_actions: int = 6):
        super().__init__()
        hidden = hidden or [64, 32]
        in_dim = state_dim + n_smell_actions
        self.net = _mlp([in_dim] + hidden + [n_actions])
        self.n_smell = n_smell_actions
        logger.debug(f"[LowerQNet] dims={[in_dim]+hidden+[n_actions]}")

    def forward(self, state: torch.Tensor, a1_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, a1_onehot], dim=-1))


class FlatQNetwork(nn.Module):
    """Flat DQN baseline: Q(s) → Q-values for 30 combined actions (5×6)."""

    def __init__(self, state_dim: int = 128, hidden: List[int] = None, n_actions: int = 30):
        super().__init__()
        hidden = hidden or [64, 32]
        self.net = _mlp([state_dim] + hidden + [n_actions])
        self.n_smell   = 5
        self.n_pattern = 6
        logger.debug(f"[FlatQNet] dims={[state_dim]+hidden+[n_actions]}")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def decode(self, flat_action: int) -> Tuple[int, int]:
        """Decode flat action index → (smell_idx, pattern_idx)."""
        return divmod(flat_action, self.n_pattern)


# ──────────────────────────── GCN Classifier (pre-training) ────────────────

class GCNClassifier(nn.Module):
    """GCN encoder + linear head for supervised pre-training."""

    def __init__(self, gcn: GCNEncoder, n_classes: int = 5):
        super().__init__()
        self.gcn  = gcn
        self.head = nn.Linear(gcn.out_ch, n_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        emb = self.gcn(x, edge_index)
        return self.head(emb)


# ──────────────────────────── Hierarchical DQN Agent ───────────────────────

class HierarchicalDQNAgent(nn.Module):
    """
    Full SmellRL agent.

    Networks:
      gcn        — shared feature extractor (pre-trained, optionally fine-tuned)
      q1         — upper Q-network (smell detection)
      q1_target  — target copy of q1
      q2         — lower Q-network (pattern recommendation)
      q2_target  — target copy of q2

    Action selection:
      select_action(graph, epsilon) → (a1, a2, state_vec)

    Training:
      train_step(batch) → (loss_q1, loss_q2)

    Target sync:
      sync_targets()
    """

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__()
        gcn_cfg = cfg["gcn"]
        dqn_cfg = cfg["dqn"]
        s_cfg   = cfg["smells"]
        p_cfg   = cfg["patterns"]

        self.device      = device
        self.gamma       = cfg["training"]["gamma"]
        self.freeze_gcn  = cfg["training"].get("freeze_gcn", False)

        # ── Networks ──────────────────────────────────────────────────────
        self.gcn = GCNEncoder(
            in_ch  = gcn_cfg["node_feature_dim"],
            hidden = gcn_cfg["hidden_dim"],
            out_ch = gcn_cfg["output_dim"],
            dropout= gcn_cfg["dropout"],
        )
        n_s = s_cfg["n_classes"]
        n_p = p_cfg["n_classes"]
        s_d = gcn_cfg["output_dim"]

        self.q1 = UpperQNetwork(s_d, dqn_cfg["upper_hidden"], n_s)
        self.q2 = LowerQNetwork(s_d, n_s, dqn_cfg["lower_hidden"], n_p)

        self.q1_target = UpperQNetwork(s_d, dqn_cfg["upper_hidden"], n_s)
        self.q2_target = LowerQNetwork(s_d, n_s, dqn_cfg["lower_hidden"], n_p)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        for net in [self.q1_target, self.q2_target]:
            for p in net.parameters():
                p.requires_grad = False

        if self.freeze_gcn:
            for p in self.gcn.parameters():
                p.requires_grad = False

        self.to(device)
        self.n_s = n_s
        self.n_p = n_p
        logger.info(f"[Agent] HierarchicalDQN ready. device={device} freeze_gcn={self.freeze_gcn}")

    # ── Utility ───────────────────────────────────────────────────────────

    def _encode(self, graph: dict) -> torch.Tensor:
        """Graph dict → [1, state_dim] tensor on device."""
        x          = graph["x"].to(self.device)
        edge_index = graph["edge_index"].to(self.device)
        with torch.set_grad_enabled(not self.freeze_gcn):
            return self.gcn(x, edge_index)          # [1, state_dim]

    def _a1_onehot(self, a1: int) -> torch.Tensor:
        oh = torch.zeros(1, self.n_s, device=self.device)
        oh[0, a1] = 1.0
        return oh

    # ── Action selection ──────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, graph: dict, epsilon: float) -> Tuple[int, int, torch.Tensor]:
        """ε-greedy action selection. Returns (a1, a2, state_vec)."""
        state = self._encode(graph)

        if np.random.random() < epsilon:
            a1 = np.random.randint(0, self.n_s)
            a2 = np.random.randint(0, self.n_p)
        else:
            q1_vals = self.q1(state)
            a1 = int(q1_vals.argmax(dim=1).item())
            oh = self._a1_onehot(a1)
            q2_vals = self.q2(state, oh)
            a2 = int(q2_vals.argmax(dim=1).item())

        return a1, a2, state.detach()

    # ── Training step ─────────────────────────────────────────────────────

    def train_step(
        self,
        batch: List[dict],
        optimizer_q1: torch.optim.Optimizer,
        optimizer_q2: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        """One gradient update from a list of transition dicts."""
        states   = torch.cat([t["state"]   for t in batch], dim=0)   # [B, state_dim]
        a1s      = torch.tensor([t["a1"]   for t in batch], device=self.device)  # [B]
        a2s      = torch.tensor([t["a2"]   for t in batch], device=self.device)
        rewards  = torch.tensor([t["reward"] for t in batch], device=self.device, dtype=torch.float32)
        # next states (same graph, terminal after one step → next_q = 0)
        # for this episodic single-step setting, we don't bootstrap next state

        # ── Q1 loss ───────────────────────────────────────────────────────
        q1_pred = self.q1(states).gather(1, a1s.unsqueeze(1)).squeeze(1)
        # target: immediate reward only (terminal transition)
        with torch.no_grad():
            q1_target = rewards  # single-step
        loss_q1 = F.smooth_l1_loss(q1_pred, q1_target)

        optimizer_q1.zero_grad()
        loss_q1.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.gcn.parameters()), 1.0)
        optimizer_q1.step()

        # ── Q2 loss ───────────────────────────────────────────────────────
        oh_a1 = torch.zeros(len(batch), self.n_s, device=self.device)
        oh_a1.scatter_(1, a1s.unsqueeze(1), 1.0)
        q2_pred = self.q2(states.detach(), oh_a1).gather(1, a2s.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q2_target = rewards
        loss_q2 = F.smooth_l1_loss(q2_pred, q2_target)

        optimizer_q2.zero_grad()
        loss_q2.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        optimizer_q2.step()

        return loss_q1.item(), loss_q2.item()

    # ── Target network sync ───────────────────────────────────────────────

    def sync_targets(self):
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        logger.debug("[Agent] Target networks synced")


# ──────────────────────────── Flat DQN Agent (baseline) ────────────────────

class FlatDQNAgent(nn.Module):
    """
    Flat DQN baseline.  Shares the same GCN encoder.
    Outputs Q-values for all 30 combined actions (5 smells × 6 patterns).
    """

    def __init__(self, gcn: GCNEncoder, cfg: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.gamma  = cfg["training"]["gamma"]
        self.gcn    = gcn
        self.n_s = cfg["smells"]["n_classes"]
        self.n_p = cfg["patterns"]["n_classes"]
        n_total  = self.n_s * self.n_p
        state_dim = cfg["gcn"]["output_dim"]
        self.q    = FlatQNetwork(state_dim, cfg["dqn"]["upper_hidden"], n_total)
        self.q_target = FlatQNetwork(state_dim, cfg["dqn"]["upper_hidden"], n_total)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False
        self.to(device)
        logger.info(f"[FlatDQN] Ready. device={device}")

    @torch.no_grad()
    def select_action(self, graph: dict, epsilon: float) -> Tuple[int, int]:
        x          = graph["x"].to(self.device)
        edge_index = graph["edge_index"].to(self.device)
        if np.random.random() < epsilon:
            flat = np.random.randint(0, self.n_s * self.n_p)
        else:
            state = self.gcn(x, edge_index)
            flat  = int(self.q(state).argmax(dim=1).item())
        return self.q.decode(flat)

    def train_step(self, batch: List[dict], optimizer: torch.optim.Optimizer) -> float:
        x_list  = [self.gcn(b["x"].to(self.device), b["edge_index"].to(self.device)) for b in batch]
        states  = torch.cat(x_list, dim=0)
        flats   = torch.tensor(
            [b["a1"] * self.n_p + b["a2"] for b in batch], device=self.device
        )
        rewards = torch.tensor([b["reward"] for b in batch], device=self.device, dtype=torch.float32)
        q_pred  = self.q(states).gather(1, flats.unsqueeze(1)).squeeze(1)
        loss    = F.smooth_l1_loss(q_pred, rewards)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())
