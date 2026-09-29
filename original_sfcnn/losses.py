"""Loss functions for the residual SFCNN channel estimator.

Tensor convention
-----------------
All network tensors have shape ``[batch, channels, N_B, N_U]``.
The four output/target channels are ordered as::

    real(H_k), imag(H_k), real(H_{k+1}), imag(H_{k+1})

The model predicts a residual.  The first four input channels contain the
reference TE estimate, so the normalized reconstructed channel is::

    reconstructed_channel = model_input[:, :4] + predicted_residual

The morph-invariance loss must compare channels expressed with the same
physical scaling.  If paired examples use different ``scale_c`` values,
unnormalize each reconstruction before calling ``morph_invariance_loss``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_LOSSES = (
    "residual_mse",
    "channel_nmse",
    "hybrid",
    "channel_nmse_cvar",
)


def _validate_core_tensors(
    prediction: torch.Tensor,
    target_delta: torch.Tensor,
    model_input: torch.Tensor,
) -> None:
    if prediction.ndim != 4:
        raise ValueError("prediction must have shape [batch, 4, N_B, N_U]")
    if prediction.shape != target_delta.shape:
        raise ValueError(
            "prediction and target_delta must have identical shapes; "
            f"got {tuple(prediction.shape)} and {tuple(target_delta.shape)}"
        )
    if prediction.shape[1] != 4:
        raise ValueError("prediction and target_delta must have four channels")
    if model_input.ndim != 4 or model_input.shape[0] != prediction.shape[0]:
        raise ValueError("model_input must be a batched four-dimensional tensor")
    if model_input.shape[1] < 4:
        raise ValueError("model_input must contain at least one four-channel view")
    if model_input.shape[2:] != prediction.shape[2:]:
        raise ValueError("model_input and prediction must have the same spatial size")


def reference_channel(model_input: torch.Tensor) -> torch.Tensor:
    """Return the first deformation view's normalized TE channel pair."""
    if model_input.ndim != 4 or model_input.shape[1] < 4:
        raise ValueError("model_input must have shape [batch, >=4, N_B, N_U]")
    return model_input[:, :4]


def reconstructed_channel(
    prediction: torch.Tensor,
    model_input: torch.Tensor,
) -> torch.Tensor:
    """Combine the reference TE estimate with the predicted residual."""
    return reference_channel(model_input) + prediction


def target_channel(
    target_delta: torch.Tensor,
    model_input: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the normalized ground-truth channel from its residual."""
    return reference_channel(model_input) + target_delta


def residual_mse_loss(
    prediction: torch.Tensor,
    target_delta: torch.Tensor,
) -> torch.Tensor:
    """Original notebook objective: MSE between predicted and true residual."""
    return F.mse_loss(prediction, target_delta)


def per_sample_channel_nmse(
    prediction: torch.Tensor,
    target_delta: torch.Tensor,
    model_input: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return one reconstructed-channel NMSE value per batch item."""
    _validate_core_tensors(prediction, target_delta, model_input)
    estimate = reconstructed_channel(prediction, model_input)
    target = target_channel(target_delta, model_input)
    reduction_dims = tuple(range(1, target.ndim))
    squared_error = (estimate - target).square().sum(dim=reduction_dims)
    target_energy = target.square().sum(dim=reduction_dims)
    return squared_error / target_energy.clamp_min(eps)


def channel_nmse_loss(
    prediction: torch.Tensor,
    target_delta: torch.Tensor,
    model_input: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean NMSE of the final reconstructed two-subcarrier channel."""
    return per_sample_channel_nmse(
        prediction, target_delta, model_input, eps=eps
    ).mean()


def hybrid_channel_loss(
    prediction: torch.Tensor,
    target_delta: torch.Tensor,
    model_input: torch.Tensor,
    residual_weight: float = 0.10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Channel NMSE plus a small residual-MSE stabilization term."""
    return channel_nmse_loss(
        prediction, target_delta, model_input, eps=eps
    ) + residual_weight * residual_mse_loss(prediction, target_delta)


def cvar_loss(
    losses: torch.Tensor,
    worst_fraction: float = 0.20,
) -> torch.Tensor:
    """Mean of the worst fraction of scalar losses (a CVaR-style objective)."""
    losses = losses.reshape(-1)
    if losses.numel() == 0:
        raise ValueError("losses cannot be empty")
    if not 0.0 < worst_fraction <= 1.0:
        raise ValueError("worst_fraction must be in (0, 1]")
    count = max(1, math.ceil(worst_fraction * losses.numel()))
    return torch.topk(losses, k=count, largest=True).values.mean()


def range_cvar_loss(
    per_sample_losses: torch.Tensor,
    range_bins: torch.Tensor,
    worst_fraction: float = 0.20,
) -> torch.Tensor:
    """CVaR over mean losses of morphing-range bins.

    ``range_bins`` contains an integer bin ID for every batch item.  This is
    the genuinely range-aware version; ordinary ``cvar_loss`` emphasizes hard
    samples but does not know their morphing amplitudes.
    """
    per_sample_losses = per_sample_losses.reshape(-1)
    range_bins = range_bins.reshape(-1)
    if per_sample_losses.shape != range_bins.shape:
        raise ValueError("per_sample_losses and range_bins must have equal length")

    bin_means = []
    for bin_id in torch.unique(range_bins):
        selected = per_sample_losses[range_bins == bin_id]
        if selected.numel() > 0:
            bin_means.append(selected.mean())
    if not bin_means:
        raise ValueError("range_bins did not contain any samples")
    return cvar_loss(torch.stack(bin_means), worst_fraction=worst_fraction)


def unnormalize_channel(
    normalized_channel: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Return a channel pair to physical scale using its saved ``scale_c``."""
    if normalized_channel.ndim != 4:
        raise ValueError("normalized_channel must be four-dimensional")
    scale = torch.as_tensor(
        scale,
        dtype=normalized_channel.dtype,
        device=normalized_channel.device,
    ).reshape(-1, 1, 1, 1)
    if scale.shape[0] != normalized_channel.shape[0]:
        raise ValueError("one scale value is required per batch item")
    return normalized_channel * scale


def morph_invariance_loss(
    channel_a: torch.Tensor,
    channel_b: torch.Tensor,
    common_target: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Require paired morphing observations to recover the same channel.

    ``channel_a`` and ``channel_b`` must represent the same propagation scene
    at two morphing amplitudes and must already be expressed on a common
    physical scale.  Passing independently normalized channels is incorrect.
    """
    if channel_a.shape != channel_b.shape or channel_a.ndim != 4:
        raise ValueError("paired channels must have identical four-dimensional shapes")

    dims = tuple(range(1, channel_a.ndim))
    difference_energy = (channel_a - channel_b).square().sum(dim=dims)

    if common_target is not None:
        if common_target.shape != channel_a.shape:
            raise ValueError("common_target must have the same shape as the channels")
        denominator = common_target.square().sum(dim=dims)
    else:
        denominator = 0.5 * (
            channel_a.square().sum(dim=dims)
            + channel_b.square().sum(dim=dims)
        )

    return (difference_energy / denominator.clamp_min(eps)).mean()


def paired_morph_invariance_loss(
    prediction_a: torch.Tensor,
    model_input_a: torch.Tensor,
    scale_a: torch.Tensor,
    prediction_b: torch.Tensor,
    model_input_b: torch.Tensor,
    scale_b: torch.Tensor,
    common_target: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convenience wrapper for paired residual-SFCNN predictions."""
    channel_a = unnormalize_channel(
        reconstructed_channel(prediction_a, model_input_a), scale_a
    )
    channel_b = unnormalize_channel(
        reconstructed_channel(prediction_b, model_input_b), scale_b
    )
    return morph_invariance_loss(
        channel_a, channel_b, common_target=common_target, eps=eps
    )


class SFCNNLoss(nn.Module):
    """Selectable loss for the existing, unpaired SFCNN training loop.

    Available names:

    - ``residual_mse``: the original notebook loss;
    - ``channel_nmse``: directly optimizes reconstructed-channel NMSE;
    - ``hybrid``: channel NMSE plus residual MSE;
    - ``channel_nmse_cvar``: mean channel NMSE plus worst-sample CVaR.

    Paired morph-invariance and true range-bin CVaR require extra dataset
    outputs, so use the corresponding standalone functions above when that
    paired dataset is introduced.
    """

    def __init__(
        self,
        name: str = "channel_nmse",
        residual_weight: float = 0.10,
        cvar_weight: float = 0.10,
        worst_fraction: float = 0.20,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if name not in VALID_LOSSES:
            raise ValueError(f"Unknown loss {name!r}; choose one of {VALID_LOSSES}")
        if residual_weight < 0.0 or cvar_weight < 0.0:
            raise ValueError("loss weights must be nonnegative")
        if not 0.0 < worst_fraction <= 1.0:
            raise ValueError("worst_fraction must be in (0, 1]")
        self.name = name
        self.residual_weight = residual_weight
        self.cvar_weight = cvar_weight
        self.worst_fraction = worst_fraction
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target_delta: torch.Tensor,
        model_input: torch.Tensor,
    ) -> torch.Tensor:
        if self.name == "residual_mse":
            return residual_mse_loss(prediction, target_delta)

        sample_nmse = per_sample_channel_nmse(
            prediction, target_delta, model_input, eps=self.eps
        )
        mean_nmse = sample_nmse.mean()

        if self.name == "channel_nmse":
            return mean_nmse
        if self.name == "hybrid":
            return mean_nmse + self.residual_weight * residual_mse_loss(
                prediction, target_delta
            )
        if self.name == "channel_nmse_cvar":
            return mean_nmse + self.cvar_weight * cvar_loss(
                sample_nmse, worst_fraction=self.worst_fraction
            )

        raise RuntimeError("unreachable loss selection")
