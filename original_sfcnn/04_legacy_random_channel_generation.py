"""Exported verbatim from notebook code cell 4."""

def generate_H_fim_deformed(P_B, Zeta_B, P_U, Zeta_U, wavelength, fs, K, L):
    """
    OFDM channel generation between two deformed single-layer FIMs.

    Returns
    -------
    H : ndarray, shape (N_U, N_B, K)
    At : ndarray, shape (N_B, L)
    Ar : ndarray, shape (N_U, L)
    AOA : ndarray, shape (L, 2)
    DOA : ndarray, shape (L, 2)
    BETA : ndarray, shape (L,)
    delay : ndarray, shape (L,)
    """

    P_B = np.asarray(P_B, dtype=float)
    Zeta_B = np.asarray(Zeta_B, dtype=float)
    P_U = np.asarray(P_U, dtype=float)
    Zeta_U = np.asarray(Zeta_U, dtype=float)

    N_B = P_B.shape[1]
    N_U = P_U.shape[1]

    # Path gains
    alpha = (
        np.random.randn(L) + 1j * np.random.randn(L)
    ) / np.sqrt(2 * L)

    # AoD, BS side
    phi_t = 2 * np.pi * np.random.rand(L) - np.pi
    theta_t = np.pi * np.random.rand(L)

    # AoA, UE side
    phi_r = 2 * np.pi * np.random.rand(L) - np.pi
    theta_r = np.pi * np.random.rand(L)

    # Delays
    tau_max = 7.8e-8
    tau_min = 7.6e-10
    delay = tau_min + (tau_max - tau_min) * np.random.rand(L)

    BETA = alpha.copy()
    AOA = np.column_stack((phi_r, theta_r))
    DOA = np.column_stack((phi_t, theta_t))

    # Steering matrices
    At = np.zeros((N_B, L), dtype=complex)
    Ar = np.zeros((N_U, L), dtype=complex)

    for ell in range(L):
        At[:, ell] = steering_vector_fim(
            P_B, Zeta_B, wavelength, phi_t[ell], theta_t[ell]
        )
        Ar[:, ell] = steering_vector_fim(
            P_U, Zeta_U, wavelength, phi_r[ell], theta_r[ell]
        )

    # OFDM channel
    H = np.zeros((N_U, N_B, K), dtype=complex)

    for k in range(K):
        Hk = np.zeros((N_U, N_B), dtype=complex)

        for ell in range(L):
            ofdm_phase = np.exp(
                -1j * 2 * np.pi * (k / K) * fs * delay[ell]
            )

            Hk += BETA[ell] * ofdm_phase * np.outer(
                Ar[:, ell],
                np.conj(At[:, ell])
            )

        H[:, :, k] = np.sqrt(N_U * N_B / L) * Hk

    return H, At, Ar, AOA, DOA, BETA, delay
