"""Exported verbatim from notebook code cell 12."""

import numpy as np


def generate_multi_pilots(N_U, Tpilots):
    """
    Generate Tpilots unitary pilot matrices.

    Observation model:
        Y_t[:,:,k] = H[:,:,k] @ S_t + N

    Therefore:
        S_t has shape (N_U, N_U)

    Parameters
    ----------
    N_U : int
        Number of UE elements/streams.
    Tpilots : int
        Number of pilot matrices.

    Returns
    -------
    S_all : ndarray, shape (N_U, N_U, Tpilots)
        Collection of unitary pilot matrices.
    """

    # --------------------------------------------------------
    # Unitary DFT matrix
    # --------------------------------------------------------
    n = np.arange(N_U)
    k = n.reshape(-1, 1)

    F = np.exp(-2j * np.pi * k * n / N_U) / np.sqrt(N_U)

    # --------------------------------------------------------
    # Pilot storage
    # --------------------------------------------------------
    S_all = np.zeros((N_U, N_U, Tpilots), dtype=complex)

    # --------------------------------------------------------
    # Generate pilots
    # --------------------------------------------------------
    for t in range(Tpilots):

        # Random unit-modulus phase diversity
        d = np.exp(1j * 2 * np.pi * np.random.rand(N_U))

        D = np.diag(d)

        S_all[:, :, t] = F @ D

    return S_all
