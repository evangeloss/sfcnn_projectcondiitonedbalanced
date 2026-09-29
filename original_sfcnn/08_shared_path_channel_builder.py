"""Exported verbatim from notebook code cell 8."""

import numpy as np


def build_H_fim_from_paths(
    params,
    P_B0, Zeta_B,
    P_U0, Zeta_U,
    wavelength, fs, K
):
    """
    Build OFDM channel with 3D steering + deformation.

    Parameters
    ----------
    params : dict
        Dictionary containing:
            params['AOA_az'] : (L,)
            params['AOA_el'] : (L,)
            params['DOA_az'] : (L,)
            params['DOA_el'] : (L,)
            params['BETA']   : (L,)
            params['delay']  : (L,)

    P_B0, P_U0 : ndarray, shape (3, N)
        Rigid coordinates.

    Zeta_B, Zeta_U : ndarray, shape (3, N)
        Deformation offsets.

    wavelength : float
        Wavelength.

    fs : float
        Sampling frequency.

    K : int
        Number of OFDM subcarriers.

    Returns
    -------
    H : ndarray, shape (N_B, N_U, K)
        OFDM channel tensor.
    """

    BETA = np.asarray(params["BETA"])

    L = len(BETA)

    N_B = P_B0.shape[1]
    N_U = P_U0.shape[1]

    H = np.zeros((N_B, N_U, K), dtype=complex)

    for k in range(K):

        Hk = np.zeros((N_B, N_U), dtype=complex)

        for l in range(L):

            # ------------------------------------------------
            # Steering vectors
            # ------------------------------------------------
            a_B = steering_vector_fim(
                P_B0,
                Zeta_B,
                wavelength,
                params["AOA_az"][l],
                params["AOA_el"][l]
            )   # (N_B,)

            a_U = steering_vector_fim(
                P_U0,
                Zeta_U,
                wavelength,
                params["DOA_az"][l],
                params["DOA_el"][l]
            )   # (N_U,)

            # ------------------------------------------------
            # OFDM phase term
            # ------------------------------------------------
            tau_l = params["delay"][l]

            phase_k = np.exp(
                -1j * 2 * np.pi * (k / K) * fs * tau_l
            )

            # ------------------------------------------------
            # Channel contribution
            # ------------------------------------------------
            Hk += (
                params["BETA"][l]
                * np.outer(a_B, np.conj(a_U))
                * phase_k
            )

        H[:, :, k] = Hk

    return H
