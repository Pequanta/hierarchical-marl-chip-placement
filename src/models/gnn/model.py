import torch
import torch.nn as nn
from .encoder import GNNEncoder
class GNNPlacementRegressor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_channels, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x, edge_index)
        return torch.sigmoid(self.head(self.dropout(embeddings)))

