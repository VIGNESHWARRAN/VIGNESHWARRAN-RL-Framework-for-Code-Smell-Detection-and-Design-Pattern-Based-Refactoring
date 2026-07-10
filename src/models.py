import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

logger = logging.getLogger("SmellRL.models")

# ──────────────────────────── Shared Pooling + Fusion ──────────────────────

class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = self.gate(h)
        weights = weights / (weights.sum() + 1e-9)
        return (weights * h).sum(dim=0, keepdim=True)


class NodeFeatureFusion(nn.Module):
    """
    Fuses structural (15 dims) and semantic (768 dims) node features into a
    balanced 128-dim representation.

    Structural branch  (15 → 104): 2-layer MLP with ReLU.
    Semantic branch   (768 → 24):  Single linear projection + LayerNorm.
    Output: cat([struct_out, sem_out]) = 128 dims per node (18.75% semantic weightage).
    """
    STRUCT_DIM = 15
    SEM_DIM    = 768
    STRUCT_OUT = 104
    SEM_OUT    = 24
    FUSED_DIM  = 128

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.struct_proj = nn.Sequential(
            nn.Linear(self.STRUCT_DIM, 48),
            nn.ReLU(),
            nn.Linear(48, self.STRUCT_OUT),
            nn.LayerNorm(self.STRUCT_OUT),
        )
        self.sem_proj = nn.Sequential(
            nn.Linear(self.SEM_DIM, self.SEM_OUT),
            nn.LayerNorm(self.SEM_OUT),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.STRUCT_DIM + self.SEM_DIM:
            raise ValueError(
                f"NodeFeatureFusion expects input width ≥ "
                f"{self.STRUCT_DIM + self.SEM_DIM} (got {x.shape[1]})."
            )
        struct = x[:, :self.STRUCT_DIM]
        sem    = x[:, self.STRUCT_DIM:self.STRUCT_DIM + self.SEM_DIM]
        struct_out = self.struct_proj(struct)
        sem_out    = self.sem_proj(sem)
        fused = torch.cat([struct_out, sem_out], dim=-1)
        return self.dropout(fused)


# ──────────────────────────── Relational Graph Attention (R-GAT) ──────────

class RGATLayer(nn.Module):
    """
    Single relational multi-head graph attention layer.
    Aggregates message-passing across distinct relation types.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_edge_types: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if out_ch % num_heads != 0:
            raise ValueError(f"out_ch ({out_ch}) must be divisible by num_heads ({num_heads})")

        self.num_heads     = num_heads
        self.num_edge_types = num_edge_types
        self.head_dim      = out_ch // num_heads
        self.out_ch        = out_ch

        self.W = nn.Parameter(torch.empty(num_edge_types, num_heads, in_ch, self.head_dim))
        self.a = nn.Parameter(torch.empty(num_edge_types, num_heads, 2 * self.head_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.attn_drop  = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.view(-1, self.W.shape[2], self.W.shape[3]))
        nn.init.xavier_uniform_(self.a.view(-1, 1, 2 * self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        N = x.size(0)

        if edge_index.numel() == 0:
            return x.new_zeros(N, self.out_ch)

        row, col = edge_index[0], edge_index[1]
        E = row.size(0)

        h_all = torch.einsum("ni,rhid->nrhd", x, self.W)

        h_src = h_all[row, edge_type]
        h_dst = h_all[col, edge_type]

        a_e   = self.a[edge_type]
        h_cat = torch.cat([h_src, h_dst], dim=-1)
        e_raw = self.leaky_relu((a_e * h_cat).sum(dim=-1))

        col_exp = col.unsqueeze(1).expand(-1, self.num_heads)

        e_max = x.new_full((N, self.num_heads), -1e9)
        e_max.scatter_reduce_(0, col_exp, e_raw, reduce="amax", include_self=True)
        e_shifted = e_raw - e_max[col]

        e_exp = e_shifted.exp()
        e_sum = x.new_zeros(N, self.num_heads)
        e_sum.scatter_add_(0, col_exp, e_exp)
        alpha = e_exp / (e_sum[col] + 1e-9)
        alpha = self.attn_drop(alpha)

        weighted = h_src * alpha.unsqueeze(-1)

        out = x.new_zeros(N, self.num_heads, self.head_dim)
        col_exp3 = (
            col.unsqueeze(1).unsqueeze(2)
            .expand(-1, self.num_heads, self.head_dim)
        )
        out.scatter_add_(0, col_exp3, weighted)

        return F.elu(out.view(N, self.out_ch))


class RGATEncoder(nn.Module):
    """
    Two-layer Relational Graph Attention encoder.
    Fuses node features, computes message passing on edges, and pools into graph embedding.
    """
    def __init__(
        self,
        in_ch: int,
        hidden: int = 128,
        out_ch: int = 128,
        num_edge_types: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion    = NodeFeatureFusion(dropout=dropout)
        fused_dim      = NodeFeatureFusion.FUSED_DIM

        self.conv1     = RGATLayer(fused_dim, hidden, num_edge_types, num_heads, dropout)
        self.norm1     = nn.LayerNorm(hidden)
        self.conv2     = RGATLayer(hidden, out_ch, num_edge_types, num_heads, dropout)
        self.norm2     = nn.LayerNorm(out_ch)
        self.skip_proj = nn.Linear(fused_dim, out_ch)
        self.dropout   = nn.Dropout(dropout)
        self.pool      = AttentionPooling(out_ch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        x  = self.fusion(x)
        h  = self.conv1(x, edge_index, edge_type)
        h  = self.norm1(h)
        h  = self.dropout(h)
        h2 = self.conv2(h, edge_index, edge_type)
        h2 = h2 + self.skip_proj(x)
        h2 = self.norm2(h2)
        return self.pool(h2)


# ──────────────────────────── DQN Agent Q-Network ───────────────────────────

class DQNClassifier(nn.Module):
    def __init__(self, state_dim: int, hidden: list, n_classes: int = 6, dropout: float = 0.2):
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
    Q-Network Smell/Refactoring Agent.
    Outputs Q-values for 6 Refactoring Actions.
    """
    def __init__(self, cfg: dict, feature_dim: int, device: torch.device):
        super().__init__()
        self.device    = device
        self.n_classes = cfg.get("patterns", {}).get("n_classes", 6)
        
        if self.n_classes != 6:
            raise ValueError(f"Expected 6 pattern actions in SmellDetectionAgent, got {self.n_classes}")

        self.gcn = RGATEncoder(
            in_ch=feature_dim,
            hidden=cfg["gcn"]["hidden_dim"],
            out_ch=cfg["gcn"]["output_dim"],
            num_edge_types=cfg["gcn"].get("num_edge_types", 3),
            num_heads=cfg["gcn"].get("num_heads", 4),
            dropout=cfg["gcn"]["dropout"],
        )

        dqn_dropout = cfg.get("dqn", {}).get("dqn_dropout", 0.2)

        self.q = DQNClassifier(
            state_dim=cfg["gcn"]["output_dim"],
            hidden=cfg["dqn"]["hidden_layers"],
            n_classes=self.n_classes,
            dropout=dqn_dropout,
        )
        self.q_target = DQNClassifier(
            state_dim=cfg["gcn"]["output_dim"],
            hidden=cfg["dqn"]["hidden_layers"],
            n_classes=self.n_classes,
            dropout=dqn_dropout,
        )
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.to(device)

    @torch.no_grad()
    def select_action(self, graph: dict, epsilon: float) -> Tuple[int, torch.Tensor]:
        x          = graph["x"].to(self.device)
        edge_index = graph["edge_index"].to(self.device)
        edge_type  = graph["edge_type"].to(self.device)
        
        state = self.gcn(x, edge_index, edge_type)

        if np.random.random() < epsilon:
            action = np.random.randint(0, self.n_classes)
        else:
            action = int(self.q(state).argmax(dim=1).item())
        return action, state.detach()

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())