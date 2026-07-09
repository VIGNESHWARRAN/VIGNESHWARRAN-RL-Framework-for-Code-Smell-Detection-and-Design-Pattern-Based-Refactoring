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


# ──────────────────────────── PRIMARY ENCODER: R-GAT ──────────────────────
#
# Relational Graph Attention Network layer.
#
# Key idea: standard GCN averages all neighbours equally regardless of the
# edge type connecting them.  For code-smell detection this is catastrophic:
# a CALLS edge from a method to 10 external classes is a FeatureEnvy signal,
# but a CONTAINS edge from a class to 10 methods is completely normal.
# RGATLayer learns a *separate* weight matrix W_r and attention vector a_r
# for each edge type r, so the network can assign context-sensitive importance
# to each relationship.

class RGATLayer(nn.Module):
    """
    Single relational multi-head graph attention layer.

    For each edge (i → j) of type r and each head k:
        h_i^{r,k}  =  x_i  @  W[r, k]              # type-specific transform
        e_ij^{r,k} =  LeakyReLU( a[r,k] · [h_i || h_j] )  # attention score
        α_ij^k     =  softmax over all j's in-neighbours of all types
        out_j^k    =  Σ_i  α_ij^k · h_i^{r,k}      # weighted aggregation
    Output: concat of K heads  →  [N, out_ch]

    Args:
        in_ch:          Input feature dimension (matches NodeFeatureFusion.FUSED_DIM = 128).
        out_ch:         Output dimension (must be divisible by num_heads).
        num_edge_types: Number of distinct relation types (default 3: CONTAINS/CALLS/ACCESSES_FIELD).
        num_heads:      Number of parallel attention heads (default 4).
        dropout:        Attention dropout probability.
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

        # Per-edge-type, per-head weight matrices
        # Shape: [num_edge_types, num_heads, in_ch, head_dim]
        self.W = nn.Parameter(torch.empty(num_edge_types, num_heads, in_ch, self.head_dim))

        # Per-edge-type, per-head attention vectors
        # Shape: [num_edge_types, num_heads, 2 * head_dim]
        self.a = nn.Parameter(torch.empty(num_edge_types, num_heads, 2 * self.head_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.attn_drop  = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier init for each (edge_type, head) weight block
        nn.init.xavier_uniform_(self.W.view(-1, self.W.shape[2], self.W.shape[3]))
        nn.init.xavier_uniform_(self.a.view(-1, 1, 2 * self.head_dim))

    def forward(
        self,
        x: torch.Tensor,           # [N, in_ch]
        edge_index: torch.Tensor,  # [2, E]
        edge_type: torch.Tensor,   # [E]  integer in {0, …, num_edge_types-1}
    ) -> torch.Tensor:
        N = x.size(0)

        if edge_index.numel() == 0:
            # No edges: return zero activations (skip connection in RGATEncoder handles this)
            return x.new_zeros(N, self.out_ch)

        row, col = edge_index[0], edge_index[1]   # row=source, col=destination
        E = row.size(0)

        # ── 1. Type-specific linear transforms ──────────────────────────────
        # W[edge_type]: [E, num_heads, in_ch, head_dim]
        W_e   = self.W[edge_type]                           # [E, H, in_ch, head_dim]
        x_src = x[row]                                       # [E, in_ch]
        x_dst = x[col]                                       # [E, in_ch]

        # h_src[e,k] = x_src[e] @ W_e[e,k]   →  [E, H, head_dim]
        h_src = torch.einsum("ei,ehid->ehd", x_src, W_e)   # [E, H, head_dim]
        h_dst = torch.einsum("ei,ehid->ehd", x_dst, W_e)   # [E, H, head_dim]

        # ── 2. Attention score e_ij = LeakyReLU(a · [h_src || h_dst]) ──────
        a_e   = self.a[edge_type]                            # [E, H, 2*head_dim]
        h_cat = torch.cat([h_src, h_dst], dim=-1)           # [E, H, 2*head_dim]
        e_raw = self.leaky_relu((a_e * h_cat).sum(dim=-1))  # [E, H]

        # ── 3. Numerically-stable scatter softmax per destination node ───────
        col_exp = col.unsqueeze(1).expand(-1, self.num_heads)  # [E, H]

        # Subtract per-destination max for numerical stability (PyTorch ≥ 2.0)
        e_max = x.new_full((N, self.num_heads), -1e9)
        e_max.scatter_reduce_(0, col_exp, e_raw, reduce="amax", include_self=True)
        e_shifted = e_raw - e_max[col]                         # [E, H]

        e_exp = e_shifted.exp()
        e_sum = x.new_zeros(N, self.num_heads)
        e_sum.scatter_add_(0, col_exp, e_exp)
        alpha = e_exp / (e_sum[col] + 1e-9)                   # [E, H]
        alpha = self.attn_drop(alpha)

        # ── 4. Aggregate weighted source features to destination ─────────────
        weighted = h_src * alpha.unsqueeze(-1)                 # [E, H, head_dim]

        out = x.new_zeros(N, self.num_heads, self.head_dim)
        col_exp3 = (
            col.unsqueeze(1).unsqueeze(2)
            .expand(-1, self.num_heads, self.head_dim)
        )
        out.scatter_add_(0, col_exp3, weighted)

        # ── 5. Flatten heads and activate ────────────────────────────────────
        return F.elu(out.view(N, self.out_ch))


class RGATEncoder(nn.Module):
    """
    Two-layer Relational Graph Attention encoder.

    Architecture (drop-in replacement for the legacy GCNEncoder):
        NodeFeatureFusion(783 → 128)
            ↓
        RGATLayer(128 → 128, heads=4)   ← learns per-edge-type attention
        LayerNorm + Dropout
            ↓
        RGATLayer(128 → 128, heads=4)
        + skip_proj(128 → 128)          ← residual connection
        LayerNorm
            ↓
        AttentionPooling(128 → 1×128)   ← graph-level embedding
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
        # in_ch accepted for API compatibility; NodeFeatureFusion fixes the real input dim.
        self.fusion    = NodeFeatureFusion(dropout=dropout)
        fused_dim      = NodeFeatureFusion.FUSED_DIM          # 128

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
        x  = self.fusion(x)                          # [N, raw_dim] → [N, 128]
        h  = self.conv1(x, edge_index, edge_type)
        h  = self.norm1(h)
        h  = self.dropout(h)
        h2 = self.conv2(h, edge_index, edge_type)
        h2 = h2 + self.skip_proj(x)                  # residual from fused input
        h2 = self.norm2(h2)
        return self.pool(h2)                          # [1, out_ch]


# ──────────────────────────── PRIMARY DETECTOR ─────────────────────────────

class SupervisedSmellDetector(nn.Module):
    """
    Primary inference model: wraps RGATEncoder + linear classifier.

    Trained end-to-end with Focal Loss via SupervisedTrainer (training.py).
    Exposes the same select_action() interface as the legacy SmellDetectionAgent
    so ExperimentRunner works with both without modification.
    """
    def __init__(
        self,
        encoder: RGATEncoder,
        classifier: nn.Linear,
        n_classes: int,
        device: torch.device,
    ):
        super().__init__()
        self.encoder    = encoder
        self.classifier = classifier
        self.n_classes  = n_classes
        self.device     = device
        self.to(device)

    @torch.no_grad()
    def select_action(self, graph: dict, epsilon: float = 0.0) -> Tuple[int, torch.Tensor]:
        """
        epsilon is accepted for interface compatibility with ExperimentRunner
        but is ignored — the supervised model always uses argmax.
        """
        x          = graph["x"].to(self.device)
        edge_index = graph["edge_index"].to(self.device)
        edge_type  = graph["edge_type"].to(self.device)
        state      = self.encoder(x, edge_index, edge_type)
        action     = int(self.classifier(state).argmax(dim=1).item())
        return action, state.detach()

    def sync_target(self):
        """No-op kept for interface compatibility with DQN ablation code."""
        pass


# ──────────────────────────── ABLATION: Legacy GCN ─────────────────────────
# Retained to reproduce the old GCN+DQN baseline numbers reported in the paper.
# Not used in the primary training pipeline.

class GCNLayer(nn.Module):
    """Legacy homogeneous GCN layer. Ablation baseline only."""
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


class GCNEncoder(nn.Module):
    """Legacy homogeneous GCN encoder. Ablation baseline only."""
    def __init__(self, in_ch: int, hidden: int = 128, out_ch: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fusion    = NodeFeatureFusion(dropout=dropout)
        fused_dim      = NodeFeatureFusion.FUSED_DIM
        self.conv1     = GCNLayer(fused_dim, hidden)
        self.norm1     = nn.LayerNorm(hidden)
        self.conv2     = GCNLayer(hidden, out_ch)
        self.norm2     = nn.LayerNorm(out_ch)
        self.skip_proj = nn.Linear(fused_dim, out_ch)
        self.dropout   = nn.Dropout(dropout)
        self.pool      = AttentionPooling(out_ch)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor = None) -> torch.Tensor:
        # edge_type is accepted but ignored — GCN is homogeneous (ablation baseline).
        # Accepting it makes GCNEncoder call-compatible with RGATEncoder.
        x  = self.fusion(x)
        h  = self.conv1(x, edge_index)
        h  = self.norm1(h)
        h  = self.dropout(h)
        h2 = self.conv2(h, edge_index)
        h2 = h2 + self.skip_proj(x)
        h2 = self.norm2(h2)
        return self.pool(h2)



# ──────────────────────────── ABLATION: DQN Classifier ─────────────────────

class DQNClassifier(nn.Module):
    """Single-phase DQN mapping GCN embedding to 5 smell classes. Ablation baseline."""
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
    DQN-based smell detection agent.

    Retained as an ablation baseline to explicitly demonstrate why the transition
    to supervised Focal Loss (SupervisedSmellDetector) was mathematically necessary.

    Supports both encoder types via config:
        gcn.encoder_type = "rgat"  →  uses RGATEncoder (edge_type-aware)
        gcn.encoder_type = "gcn"   →  uses legacy GCNEncoder (homogeneous)

    Note: Since γ=0 in the single-step formulation, this acts as a contextual
    bandit. The target network is maintained for potential future multi-step RL
    formulations but does not currently bootstrap values.
    """
    def __init__(self, cfg: dict, feature_dim: int, device: torch.device):
        super().__init__()
        self.device    = device
        self.n_classes = cfg["smells"]["n_classes"]

        encoder_type = cfg["gcn"].get("encoder_type", "rgat")
        self._encoder_type = encoder_type

        if encoder_type == "rgat":
            self.gcn = RGATEncoder(
                in_ch=feature_dim,
                hidden=cfg["gcn"]["hidden_dim"],
                out_ch=cfg["gcn"]["output_dim"],
                num_edge_types=cfg["gcn"].get("num_edge_types", 3),
                num_heads=cfg["gcn"].get("num_heads", 4),
                dropout=cfg["gcn"]["dropout"],
            )
        else:  # "gcn" — legacy homogeneous baseline
            self.gcn = GCNEncoder(
                in_ch=feature_dim,
                hidden=cfg["gcn"]["hidden_dim"],
                out_ch=cfg["gcn"]["output_dim"],
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

        if self._encoder_type == "rgat":
            edge_type = graph["edge_type"].to(self.device)
            state = self.gcn(x, edge_index, edge_type)
        else:
            state = self.gcn(x, edge_index)

        if np.random.random() < epsilon:
            action = np.random.randint(0, self.n_classes)
        else:
            action = int(self.q(state).argmax(dim=1).item())
        return action, state.detach()

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())