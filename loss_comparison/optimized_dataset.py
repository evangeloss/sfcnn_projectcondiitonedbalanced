"""Higher-diversity dataset generator for the original residual SFCNN.

This keeps the physical OFDM grid at K subcarriers but stores only a random
subset of adjacent pairs from each independent channel realization.  It also
uses an observation-only normalization and a stratified SNR schedule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Load the preserved notebook helpers into this module without executing the
# notebook's examples, plots, or training cells.
_ORIGINAL = Path(__file__).resolve().parent.parent / "original_sfcnn"
for _filename in (
    "01_path_parameter_generation.py",
    "05_steering_vector.py",
    "08_shared_path_channel_builder.py",
    "12_multi_pilot_generation.py",
    "13_pilot_transmission.py",
    "14_ridge_te_estimator.py",
):
    _path = _ORIGINAL / _filename
    exec(compile(_path.read_text(encoding="utf-8"), str(_path), "exec"), globals())


def stratified_snr_schedule(
    sample_count: int,
    snr_values: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> np.ndarray:
    """Create a shuffled SNR schedule with controlled category counts."""
    values = np.asarray(snr_values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("snr_values cannot be empty")

    if probabilities is None:
        probabilities = np.full(values.size, 1.0 / values.size)
    else:
        probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
        if probabilities.shape != values.shape:
            raise ValueError("one SNR probability is required per SNR value")
        if np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError("SNR probabilities must be nonnegative with positive sum")
        probabilities = probabilities / probabilities.sum()

    expected = sample_count * probabilities
    counts = np.floor(expected).astype(int)
    remainder = sample_count - counts.sum()
    if remainder:
        order = np.argsort(-(expected - counts))
        counts[order[:remainder]] += 1

    schedule = np.repeat(values, counts)
    np.random.shuffle(schedule)
    return schedule


def generate_optimized_multideformation_dataset(
    N_B,
    N_U,
    fc,
    fs,
    L,
    K,
    Nch,
    SNR_set,
    P_B_all,
    Zeta_B_all,
    P_U_all,
    Zeta_U_all,
    M,
    *,
    pairs_per_channel=8,
    Tpilots=1,
    snr_probabilities=None,
    normalization_mode="observation_only",
    morphing_ratio_range=None,
    base_morphing_ratio=0.02,
    morphing_anchor_ratio=None,
    morphing_anchor_probability=0.0,
    morphing_bin_edges=None,
    return_metadata=False,
):
    """Generate independent scenes with selectable pair count and normalization.

    ``observation_only`` is deployment-valid because its scale is computed only
    from the channel estimates available to the model. ``target_assisted``
    reproduces the original notebook normalization by also including the true
    channel in the scale. The latter is retained only for controlled ablations.
    """
    if Tpilots != 1:
        raise ValueError("The preserved SFCNN input layout currently requires Tpilots=1")
    if not 1 <= pairs_per_channel <= K - 1:
        raise ValueError("pairs_per_channel must be between 1 and K-1")
    if normalization_mode not in {"observation_only", "target_assisted"}:
        raise ValueError(
            "normalization_mode must be 'observation_only' or 'target_assisted'"
        )
    if base_morphing_ratio <= 0:
        raise ValueError("base_morphing_ratio must be positive")
    if morphing_ratio_range is not None:
        morphing_min, morphing_max = map(float, morphing_ratio_range)
        if morphing_min <= 0 or morphing_max < morphing_min:
            raise ValueError(
                "morphing_ratio_range must satisfy 0 < minimum <= maximum"
            )
    else:
        morphing_min = morphing_max = float(base_morphing_ratio)
    if not 0.0 <= morphing_anchor_probability <= 1.0:
        raise ValueError("morphing_anchor_probability must be in [0, 1]")
    if morphing_anchor_probability > 0.0:
        if morphing_anchor_ratio is None or morphing_anchor_ratio <= 0:
            raise ValueError(
                "a positive morphing_anchor_ratio is required when anchoring"
            )
    if morphing_bin_edges is not None:
        morphing_bin_edges = np.asarray(morphing_bin_edges, dtype=float).reshape(-1)
        if morphing_bin_edges.size < 2:
            raise ValueError("morphing_bin_edges must contain at least two edges")
        if np.any(morphing_bin_edges <= 0) or np.any(np.diff(morphing_bin_edges) <= 0):
            raise ValueError("morphing_bin_edges must be positive and increasing")
        if (
            morphing_bin_edges[0] < morphing_min
            or morphing_bin_edges[-1] > morphing_max
        ):
            raise ValueError("morphing_bin_edges must lie inside morphing_ratio_range")
    if len(P_B_all) < M or len(P_U_all) < M:
        raise ValueError("The geometry codebook contains fewer than M views")

    wavelength = 3e8 / fc
    total_samples = Nch * pairs_per_channel
    X = np.zeros((N_B, N_U, 4 * M, total_samples), dtype=np.float32)
    V = np.zeros((N_B, N_U, 4, total_samples), dtype=np.float32)
    sample_snr = np.zeros(total_samples, dtype=np.float32)
    sample_pair = np.zeros(total_samples, dtype=np.int16)
    sample_channel = np.zeros(total_samples, dtype=np.int32)
    sample_morphing_ratio = np.zeros(total_samples, dtype=np.float32)

    S_all = generate_multi_pilots(N_U, Tpilots)
    zeta_b_zero = np.zeros((3, N_B), dtype=float)
    zeta_u_zero = np.zeros((3, N_U), dtype=float)
    snr_schedule = stratified_snr_schedule(
        Nch,
        np.asarray(SNR_set),
        None if snr_probabilities is None else np.asarray(snr_probabilities),
    )
    anchor_count = int(round(Nch * morphing_anchor_probability))
    broad_count = Nch - anchor_count
    if morphing_bin_edges is None:
        broad_schedule = np.random.uniform(morphing_min, morphing_max, size=broad_count)
    else:
        bin_count = morphing_bin_edges.size - 1
        bin_ids = np.arange(broad_count) % bin_count
        np.random.shuffle(bin_ids)
        broad_schedule = np.empty(broad_count, dtype=float)
        for bin_id in range(bin_count):
            selected = bin_ids == bin_id
            broad_schedule[selected] = np.random.uniform(
                morphing_bin_edges[bin_id],
                morphing_bin_edges[bin_id + 1],
                size=int(selected.sum()),
            )
    if anchor_count:
        morphing_schedule = np.concatenate(
            [
                np.full(anchor_count, float(morphing_anchor_ratio)),
                broad_schedule,
            ]
        )
        np.random.shuffle(morphing_schedule)
    else:
        morphing_schedule = broad_schedule

    index = 0
    for channel_index in range(Nch):
        snr_db = float(snr_schedule[channel_index])
        morphing_ratio = float(morphing_schedule[channel_index])
        morphing_scale = morphing_ratio / base_morphing_ratio
        params = generate_path_parameters(L, fc, fs)
        true_channel = build_H_fim_from_paths(
            params,
            P_B_all[0],
            zeta_b_zero,
            P_U_all[0],
            zeta_u_zero,
            wavelength,
            fs,
            K,
        )

        estimates = []
        for view in range(M):
            deformed_channel = build_H_fim_from_paths(
                params,
                P_B_all[view],
                Zeta_B_all[view] * morphing_scale,
                P_U_all[view],
                Zeta_U_all[view] * morphing_scale,
                wavelength,
                fs,
                K,
            )
            received, sigma2, _ = pilot_transmission_multi(
                deformed_channel,
                N_B,
                N_U,
                S_all,
                snr_db,
                K,
            )
            estimate = np.zeros((N_B, N_U, K), dtype=complex)
            for tone in range(K):
                estimate[:, :, tone] = ridge_TE_multipilot(
                    received[:, :, tone, :],
                    S_all,
                    lambda_r=sigma2,
                )
            estimates.append(estimate)

        selected_pairs = np.random.choice(
            K - 1,
            size=pairs_per_channel,
            replace=False,
        )
        for pair_index in selected_pairs:
            h0 = true_channel[:, :, pair_index]
            h1 = true_channel[:, :, pair_index + 1]

            # Deployment-valid scale: use only quantities available from the
            # observations, never the true target channel.
            scale_values = []
            for view in range(M):
                scale_values.append(estimates[view][:, :, pair_index].reshape(-1))
                scale_values.append(estimates[view][:, :, pair_index + 1].reshape(-1))
            if normalization_mode == "target_assisted":
                scale_values.append(h0.reshape(-1))
                scale_values.append(h1.reshape(-1))
            scale = np.max(np.abs(np.concatenate(scale_values))) + 1e-8

            model_input = np.zeros((N_B, N_U, 4 * M), dtype=np.float32)
            for view in range(M):
                r0 = estimates[view][:, :, pair_index] / scale
                r1 = estimates[view][:, :, pair_index + 1] / scale
                offset = 4 * view
                model_input[:, :, offset + 0] = r0.real
                model_input[:, :, offset + 1] = r0.imag
                model_input[:, :, offset + 2] = r1.real
                model_input[:, :, offset + 3] = r1.imag

            reference0 = estimates[0][:, :, pair_index] / scale
            reference1 = estimates[0][:, :, pair_index + 1] / scale
            delta0 = h0 / scale - reference0
            delta1 = h1 / scale - reference1

            X[:, :, :, index] = model_input
            V[:, :, 0, index] = delta0.real
            V[:, :, 1, index] = delta0.imag
            V[:, :, 2, index] = delta1.real
            V[:, :, 3, index] = delta1.imag
            sample_snr[index] = snr_db
            sample_pair[index] = pair_index
            sample_channel[index] = channel_index
            sample_morphing_ratio[index] = morphing_ratio
            index += 1

    if return_metadata:
        metadata = {
            "sample_snr_db": sample_snr,
            "sample_pair_index": sample_pair,
            "sample_channel_index": sample_channel,
            "sample_morphing_ratio_b_over_lambda": sample_morphing_ratio,
            "snr_schedule": snr_schedule,
            "normalization_mode": normalization_mode,
            "morphing_ratio_range": [morphing_min, morphing_max],
            "morphing_anchor_ratio": morphing_anchor_ratio,
            "morphing_anchor_probability": morphing_anchor_probability,
            "morphing_bin_edges": (
                None if morphing_bin_edges is None else morphing_bin_edges.tolist()
            ),
        }
        return X, V, metadata
    return X, V
