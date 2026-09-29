"""Exported verbatim from notebook code cell 13."""

def pilot_transmission_multi(H, N_B, N_U, S_all, SNR_dB, K):
    """
    Multi-pilot transmission.
    """

    # --------------------------------------------------------
    # Dimension checks
    # --------------------------------------------------------
    assert H.shape[0] == N_B, "H first dimension mismatch."
    assert H.shape[1] == N_U, "H second dimension mismatch."
    assert H.shape[2] == K,   "H subcarrier dimension mismatch."

    Tpilots = S_all.shape[2]

    # --------------------------------------------------------
    # Allocate output
    # --------------------------------------------------------
    Y_clean = np.zeros(
        (N_B, N_U, K, Tpilots),
        dtype=complex
    )

    # --------------------------------------------------------
    # Pilot transmission
    # --------------------------------------------------------
    for t in range(Tpilots):

        S = S_all[:, :, t]

        assert S.shape == (N_U, N_U), \
            "Each pilot matrix must be N_U x N_U."

        for k in range(K):

            Y_clean[:, :, k, t] = H[:, :, k] @ S

    # --------------------------------------------------------
    # Noise power
    # --------------------------------------------------------
    sigPow = np.mean(np.abs(Y_clean) ** 2)

    snrLin = 10 ** (SNR_dB / 10)

    sigma2 = sigPow / snrLin

    # --------------------------------------------------------
    # Complex Gaussian noise
    # --------------------------------------------------------
    noise = np.sqrt(sigma2 / 2) * (
        np.random.randn(*Y_clean.shape)
        + 1j * np.random.randn(*Y_clean.shape)
    )

    # --------------------------------------------------------
    # Received signal
    # --------------------------------------------------------
    Y = Y_clean + noise

    return Y, sigma2, Y_clean
