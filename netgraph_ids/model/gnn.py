"""
NetGraph-IDS — GraphSAGE node classifier for intrusion detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, SAGEConv


class NetGraphGNN(nn.Module):
    """Three-layer GraphSAGE node classifier."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)

        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)

        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.bn3 = BatchNorm(hidden_dim // 2)

        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        return self.classifier(x)

    def predict_proba(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Return softmax probabilities. Shape: [N, 2]."""
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return F.softmax(logits, dim=-1)

    def predict(
        self, x: torch.Tensor, edge_index: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Return binary predictions. Shape: [N]."""
        proba = self.predict_proba(x, edge_index)
        return (proba[:, 1] >= threshold).long()
