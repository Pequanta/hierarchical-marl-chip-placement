from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    from .layers import GraphReadout, PlacementGNNLayer, PoolingType
    from .message_passing import ActivationType, ConvolutionType, NormalizationType
except ImportError:  # pragma: no cover
    from layers import GraphReadout, PlacementGNNLayer, PoolingType
    from message_passing import ActivationType, ConvolutionType, NormalizationType


@dataclass
class GNNRepresentation:
    node_embeddings: torch.Tensor
    graph_embedding: torch.Tensor


class GNNEncoder(nn.Module):
    """Placement graph encoder that returns node-level and graph-level representations.

    The default `forward` return remains node embeddings so older callers that do
    `encoder(x, edge_index)` continue to work. RL code should call `represent`.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        conv_type: ConvolutionType = "sage",
        activation: ActivationType = "relu",
        normalization: NormalizationType = "layer",
        residual: bool = True,
        pooling: PoolingType = "mean",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)
        self.pooling = pooling

        channels = [in_channels]
        if num_layers == 1:
            channels.append(out_channels)
        else:
            channels.extend([hidden_channels] * (num_layers - 1))
            channels.append(out_channels)

        self.layers = nn.ModuleList(
            PlacementGNNLayer(
                channels[idx],
                channels[idx + 1],
                dropout=dropout,
                conv_type=conv_type,
                activation=activation,
                normalization=normalization,
                residual=residual,
            )
            for idx in range(num_layers)
        )
        self.readout = GraphReadout(pooling=pooling)

        # Backward-compatible attribute names for older code/tests.
        self.conv1 = self.layers[0]
        self.conv2 = self.layers[1] if len(self.layers) > 1 else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode_nodes(x, edge_index, batch=batch)

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        del batch
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def represent(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> GNNRepresentation:
        node_embeddings = self.encode_nodes(x, edge_index, batch=batch)
        graph_embedding = self.readout(node_embeddings, batch=batch)
        return GNNRepresentation(node_embeddings=node_embeddings, graph_embedding=graph_embedding)

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        return self.represent(x, edge_index, batch=batch).graph_embedding


class FlatPlacementObservationEncoder(nn.Module):
    """Encodes flattened placement observations with a dynamically supplied graph topology.

    Environments expose observations as `[pos_x, pos_y, features...]` repeated for every
    macro. This module reconstructs node features, runs the GNN, and returns representations
    suitable for actor/critic heads.

    Unlike the previous version, `num_macros` and `edge_index` are **not** stored as fixed
    model state. `num_macros` is inferred from the observation shape at runtime, and
    `edge_index` is passed as an argument to `forward()`. This makes the same model weights
    applicable to designs with different macro counts and connectivity patterns.

    Args:
        features_per_macro: Number of features per node (fixed across all designs).
        num_macros: Accepted for backward compatibility but ignored — use observation shape.
        edge_index: Accepted for backward compatibility but ignored — pass to forward().
    """

    def __init__(
        self,
        features_per_macro: int,
        hidden_channels: int = 128,
        out_channels: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        conv_type: ConvolutionType = "sage",
        pooling: PoolingType = "mean",
        num_macros: int | None = None,
        edge_index: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        del num_macros, edge_index  # accepted for backward compat; topology is passed to forward()
        self.features_per_macro = int(features_per_macro)
        self.encoder = GNNEncoder(
            in_channels=features_per_macro,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            conv_type=conv_type,
            pooling=pooling,
        )

    @property
    def output_dim(self) -> int:
        return self.encoder.out_channels

    def observation_to_nodes(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        if observations.shape[-1] % self.features_per_macro != 0:
            raise ValueError(
                f"Observation dim {observations.shape[-1]} is not divisible by "
                f"features_per_macro {self.features_per_macro}."
            )
        num_macros = observations.shape[-1] // self.features_per_macro
        return observations.reshape(observations.shape[0], num_macros, self.features_per_macro)

    def forward(self, observations: torch.Tensor, edge_index: torch.Tensor) -> GNNRepresentation:
        """Encode observations using the provided graph topology.

        Args:
            observations: Flattened node features of shape (batch, num_macros * features_per_macro).
            edge_index: Graph connectivity for this design, shape (2, num_edges).
        """
        node_features = self.observation_to_nodes(observations)
        node_embeddings = []
        graph_embeddings = []
        for graph_features in node_features:
            representation = self.encoder.represent(graph_features, edge_index)
            node_embeddings.append(representation.node_embeddings)
            graph_embeddings.append(representation.graph_embedding.squeeze(0))
        return GNNRepresentation(
            node_embeddings=torch.stack(node_embeddings, dim=0),
            graph_embedding=torch.stack(graph_embeddings, dim=0),
        )
