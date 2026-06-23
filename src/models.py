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

class GCNEncoder(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 128, out_ch: int = 128, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNLayer(in_ch, hidden)
        self.conv2 = GCNLayer(hidden, out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h.mean(dim=0, keepdim=True)

class DQNClassifier(nn.Module):
    """Single-phase DQN mapping GCN embedding to 5 smell classes."""
    def __init__(self, state_dim: int, hidden: list, n_classes: int = 5):
        super().__init__()
        layers = []
        dims = [state_dim] + hidden + [n_classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class SmellDetectionAgent(nn.Module):
    """Wraps GCN and DQN with target network logic."""
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
        
        self.q = DQNClassifier(
            state_dim=cfg["gcn"]["output_dim"],
            hidden=cfg["dqn"]["hidden_layers"],
            n_classes=self.n_classes
        )
        self.q_target = DQNClassifier(
            state_dim=cfg["gcn"]["output_dim"],
            hidden=cfg["dqn"]["hidden_layers"],
            n_classes=self.n_classes
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