"""Morph-conditioned SFCNN initialized from the original M-view network."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MorphConditionedSFCNN(nn.Module):
    """Residual SFCNN with FiLM conditioning on the known morphing ratio.

    The five convolution blocks match the original SFCNN. A small MLP encodes
    log((b/lambda) / 0.02), and zero-initialized FiLM heads modulate every
    block. At initialization, FiLM is exactly the identity transformation.
    """

    def __init__(self, m_views: int = 8, embedding_size: int = 32) -> None:
        super().__init__()
        if m_views <= 0:
            raise ValueError("m_views must be positive")
        self.m_views = m_views
        self.embedding_size = embedding_size
        self.reference_ratio = 0.02

        channels = [4 * m_views, 64, 64, 64, 64, 64]
        self.convolutions = nn.ModuleList(
            nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1)
            for i in range(5)
        )
        self.normalizations = nn.ModuleList(nn.BatchNorm2d(64) for _ in range(5))
        self.activation = nn.ReLU()
        self.output_layer = nn.Conv2d(64, 4, kernel_size=1)

        self.morph_encoder = nn.Sequential(
            nn.Linear(1, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
        )
        self.film_layers = nn.ModuleList(
            nn.Linear(embedding_size, 128) for _ in range(5)
        )
        for layer in self.film_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def encode_ratio(self, morphing_ratio: torch.Tensor) -> torch.Tensor:
        ratio = torch.as_tensor(
            morphing_ratio,
            dtype=self.convolutions[0].weight.dtype,
            device=self.convolutions[0].weight.device,
        ).reshape(-1, 1)
        if torch.any(ratio <= 0):
            raise ValueError("morphing_ratio must be positive")
        log_ratio = torch.log(ratio / self.reference_ratio) / math.log(10.0)
        return self.morph_encoder(log_ratio)

    def forward(
        self,
        observations: torch.Tensor,
        morphing_ratio: torch.Tensor,
    ) -> torch.Tensor:
        if observations.ndim != 4:
            raise ValueError("observations must have shape [batch, 4*M, H, W]")
        if observations.shape[1] != 4 * self.m_views:
            raise ValueError(
                f"expected {4 * self.m_views} input channels, "
                f"received {observations.shape[1]}"
            )
        embedding = self.encode_ratio(morphing_ratio)
        if embedding.shape[0] != observations.shape[0]:
            raise ValueError("one morphing ratio is required per batch item")

        features = observations
        for convolution, normalization, film in zip(
            self.convolutions, self.normalizations, self.film_layers
        ):
            features = normalization(convolution(features))
            gamma, beta = film(embedding).chunk(2, dim=1)
            features = features * (1.0 + gamma[:, :, None, None])
            features = features + beta[:, :, None, None]
            features = self.activation(features)
        return self.output_layer(features)

    def conditioning_parameters(self):
        yield from self.morph_encoder.parameters()
        yield from self.film_layers.parameters()

    def base_parameters(self):
        yield from self.convolutions.parameters()
        yield from self.normalizations.parameters()
        yield from self.output_layer.parameters()


def load_original_sfcnn_weights(
    model: MorphConditionedSFCNN,
    original_state: dict[str, torch.Tensor],
) -> None:
    """Map the original Sequential SFCNN weights into the conditioned model."""
    for block in range(5):
        old_offset = 3 * block
        convolution = model.convolutions[block]
        normalization = model.normalizations[block]
        convolution.weight.data.copy_(original_state[f"network.{old_offset}.weight"])
        convolution.bias.data.copy_(original_state[f"network.{old_offset}.bias"])
        normalization.weight.data.copy_(
            original_state[f"network.{old_offset + 1}.weight"]
        )
        normalization.bias.data.copy_(
            original_state[f"network.{old_offset + 1}.bias"]
        )
        normalization.running_mean.data.copy_(
            original_state[f"network.{old_offset + 1}.running_mean"]
        )
        normalization.running_var.data.copy_(
            original_state[f"network.{old_offset + 1}.running_var"]
        )
        normalization.num_batches_tracked.data.copy_(
            original_state[f"network.{old_offset + 1}.num_batches_tracked"]
        )
    model.output_layer.weight.data.copy_(original_state["network.15.weight"])
    model.output_layer.bias.data.copy_(original_state["network.15.bias"])
