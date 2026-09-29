"""Exported verbatim from notebook code cell 14."""

import numpy as np


def ridge_TE_multipilot(Y, S_all, lambda_r):
    """
    Ridge-regularized Tentative Estimation (TE)
    """

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------
    Nr, Tobs, Tpilots = Y.shape

    Nt = S_all.shape[0]

    # --------------------------------------------------------
    # Stack all pilot observations
    # --------------------------------------------------------
    Y_stack = np.zeros(
        (Nr, Tobs * Tpilots),
        dtype=complex
    )

    S_stack = np.zeros(
        (Nt, Tobs * Tpilots),
        dtype=complex
    )

    for t in range(Tpilots):

        idx_start = t * Tobs
        idx_end = (t + 1) * Tobs

        Y_stack[:, idx_start:idx_end] = Y[:, :, t]
        S_stack[:, idx_start:idx_end] = S_all[:, :, t]

    # --------------------------------------------------------
    # Ridge estimator
    # R = Y S^H (S S^H + λI)^(-1)
    # --------------------------------------------------------
    gram = S_stack @ np.conj(S_stack.T)

    R = (
        Y_stack
        @ np.conj(S_stack.T)
        @ np.linalg.inv(
            gram + lambda_r * np.eye(Nt)
        )
    )

    return R
