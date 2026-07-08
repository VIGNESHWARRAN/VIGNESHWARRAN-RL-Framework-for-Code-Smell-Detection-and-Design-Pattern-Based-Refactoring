import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

logger = logging.getLogger("SmellRL.models")

class GCNLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return F.relu(self.linear(x))
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(x.size(0), device=x.device).scatter_add(0, row, torch.ones(row.size(0), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e6)
        deg_inv_sqrt[deg == 0] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), x[row] * norm.unsqueeze(1))
        agg = agg + x
        return F.relu(self.linear(agg))

class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = self.gate(h)
        weights = weights / (weights.sum() + 1e-9)
        return (weights * h).sum(dim=0, keepdim=True)

# ──────────────────────────── Node Feature Fusion ─────────────────────────

class NodeFeatureFusion(nn.Module):
    """
    Fuses structural (15 dims) and semantic (768 dims) node features into a
    balanced 128-dim representation, giving structural features 3× more
    representational capacity than semantic embeddings (96 vs 32 dims).

    Without this, the raw concatenated vector [15 | 768] = 783 dims causes
    the GCN's first linear layer to receive 98.1% semantic signal, effectively
    drowning out the hand-crafted CK and data-usage features.

    Structural branch  (15 → 96):  2-layer MLP with ReLU — learns non-linear
        interactions across node-type, CK metrics, and data-usage features.
    Semantic branch   (768 → 32):  Single linear projection + LayerNorm —
        compresses GraphCodeBERT's redundant high-dim space to a compact slot.
    Output: cat([struct_out, sem_out]) = 128 dims per node.
    """
    STRUCT_DIM = 15    # 3 node-type + 6 CK + 6 data-usage
    SEM_DIM    = 768   # GraphCodeBERT CLS embedding
    STRUCT_OUT = 96    # 3× semantic — structure gets priority
    SEM_OUT    = 32    # compressed semantic
    FUSED_DIM  = 128   # STRUCT_OUT + SEM_OUT (must equal gcn.hidden_dim)

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        # Structural: 2-layer MLP to learn non-linear feature interactions
        self.struct_proj = nn.Sequential(
            nn.Linear(self.STRUCT_DIM, 48),
            nn.ReLU(),
            nn.Linear(48, self.STRUCT_OUT),
            nn.LayerNorm(self.STRUCT_OUT),
        )
        # Semantic: single linear compression; semantics already pre-trained
        self.sem_proj = nn.Sequential(
            nn.Linear(self.SEM_DIM, self.SEM_OUT),
            nn.LayerNorm(self.SEM_OUT),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 783] — structural in [:15], semantic in [15:783]
        # For semantic ablation, AblatedDataset zeroes x[:, 15:] before calling
        # forward, so sem_proj(zeros) ≈ bias-only → near-zero semantic output.
        if x.shape[1] < self.STRUCT_DIM + self.SEM_DIM:
            raise ValueError(
                f"NodeFeatureFusion expects input width ≥ "
                f"{self.STRUCT_DIM + self.SEM_DIM} (got {x.shape[1]}). "
                "For semantic ablation zero out dims "
                f"[{self.STRUCT_DIM}:] via AblatedDataset — do not slice."
            )
        struct = x[:, :self.STRUCT_DIM]                             # [N, 15]
        sem    = x[:, self.STRUCT_DIM:self.STRUCT_DIM + self.SEM_DIM]  # [N, 768]
        struct_out = self.struct_proj(struct)            # [N, 96]
        sem_out    = self.sem_proj(sem)                  # [N, 32]
        fused = torch.cat([struct_out, sem_out], dim=-1) # [N, 128]
        return self.dropout(fused)


class GCNEncoder(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 128, out_ch: int = 128, dropout: float = 0.1):
        super().__init__()
        # in_ch is accepted for API compatibility but NodeFeatureFusion handles
        # the actual input projection; GCN layers always operate on FUSED_DIM.
        self.fusion = NodeFeatureFusion(dropout=dropout)
        fused_dim = NodeFeatureFusion.FUSED_DIM  # 128
        self.conv1 = GCNLayer(fused_dim, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = GCNLayer(hidden, out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.skip_proj = nn.Linear(fused_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.pool = AttentionPooling(out_ch)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.fusion(x)              # [N, raw_dim] → [N, 128] balanced representation
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = self.dropout(h)
        h2 = self.conv2(h, edge_index)
        h2 = h2 + self.skip_proj(x)    # skip_proj: Linear(128 → out_ch)
        h2 = self.norm2(h2)
        return self.pool(h2)

class DQNClassifier(nn.Module):
    """Single-phase DQN mapping GCN embedding to 5 smell classes."""
    def __init__(self, state_dim: int, hidden: list, n_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        layers = []
        dims = [state_dim] + hidden + [n_classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class SmellDetectionAgent(nn.Module):
    """
    Wraps GCN and DQN with target network logic.
    Note: Since γ=0 in the current single-step formulation, this acts as a 
    contextual bandit. The target network is maintained for potential future 
    multi-step RL formulations but does not currently bootstrap values.
    """
    def __init__(self, cfg: dict, feature_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.n_classes = cfg["smells"]["n_classes"]
        
        self.gcn = GCNEncoder(
            in_ch=feature_dim,
            hidden=cfg["gcn"]["hidden_dim"],
            out_ch=cfg["gcn"]["output_dim"],
            dropout=cfg["gcn"]["dropout"]
        )
        
        dqn_dropout = cfg.get("dqn", {}).get("dqn_dropout", 0.2)
        
        self.q = DQNClassifier(
            state_dim=cfg["gcn"]["output_dim"],
            hidden=cfg["dqn"]["hidden_layers"],
            n_classes=self.n_classes,
            dropout=dqn_dropout
        )
        self.q_target = DQNClassifier(
            state_dim=cfg["gcn"]["output_dim"],
            hidden=cfg["dqn"]["hidden_layers"],
            n_classes=self.n_classes,
            dropout=dqn_dropout
        )
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False
            
        self.to(device)

    @torch.no_grad()
    def select_action(self, graph: dict, epsilon: float) -> Tuple[int, torch.Tensor]:
        x = graph["x"].to(self.device)
        edge_index = graph["edge_index"].to(self.device)
        state = self.gcn(x, edge_index)
        
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.n_classes)
        else:
            action = int(self.q(state).argmax(dim=1).item())
        return action, state.detach()

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())