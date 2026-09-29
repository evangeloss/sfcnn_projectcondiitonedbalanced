"""Exported verbatim from notebook code cell 15."""

def generate_Dataset_multiDef_multipilot(
    N_B, N_U, fc, fs, L, K, Nch, SNR_set,
    P_B_all, Zeta_B_all, P_U_all, Zeta_U_all,
    M, Tpilots=1
):
    """
    Multi-deformation + multi-pilot residual dataset generator.

    Outputs
    -------
    X : ndarray, shape (N_B, N_U, 4*M, Nch*(K-1)), dtype float32
        CNN input.

    V : ndarray, shape (N_B, N_U, 4, Nch*(K-1)), dtype float32
        Residual CNN target.

    Notes
    -----
    Input uses all M deformation views.
    Target uses undeformed reference channel from deformation m = 0 geometry.
    """

    numPairs = K - 1
    wavelength = 3e8 / fc

    X = np.zeros((N_B, N_U, 4 * M, Nch * numPairs), dtype=np.float32)
    V = np.zeros((N_B, N_U, 4,     Nch * numPairs), dtype=np.float32)

    # --------------------------------------------------------
    # Multi-pilot codebook
    # --------------------------------------------------------
    S_all = generate_multi_pilots(N_U, Tpilots)

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------
    assert len(P_B_all) == M, "P_B_all must have M entries."
    assert len(Zeta_B_all) == M, "Zeta_B_all must have M entries."
    assert len(P_U_all) == M, "P_U_all must have M entries."
    assert len(Zeta_U_all) == M, "Zeta_U_all must have M entries."

    for m in range(M):
        assert P_B_all[m].shape == (3, N_B), f"P_B_all[{m}] wrong size."
        assert Zeta_B_all[m].shape == (3, N_B), f"Zeta_B_all[{m}] wrong size."
        assert P_U_all[m].shape == (3, N_U), f"P_U_all[{m}] wrong size."
        assert Zeta_U_all[m].shape == (3, N_U), f"Zeta_U_all[{m}] wrong size."

    # --------------------------------------------------------
    # Zero deformation matrices for true undeformed target
    # --------------------------------------------------------
    Zeta_B_zero = np.zeros((3, N_B), dtype=float)
    Zeta_U_zero = np.zeros((3, N_U), dtype=float)

    SNR_set = np.asarray(SNR_set).reshape(-1)

    idx = 0

    for n in range(Nch):

        # ----------------------------------------------------
        # Select SNR for this realization
        # ----------------------------------------------------
        if SNR_set.size == 1:
            SNR_use = float(SNR_set[0])
        else:
            SNR_use = float(np.random.choice(SNR_set))

        # ----------------------------------------------------
        # One shared propagation environment
        # ----------------------------------------------------
        params = generate_path_parameters(L, fc, fs)

        H_all = []
        R_all = []

        # ----------------------------------------------------
        # True undeformed reference target
        # MATLAB used P_B_all{1}, P_U_all{1} with zero zeta
        # ----------------------------------------------------
        H_true = build_H_fim_from_paths(
            params,
            P_B_all[0], Zeta_B_zero,
            P_U_all[0], Zeta_U_zero,
            wavelength, fs, K
        )

        assert H_true.shape == (N_B, N_U, K), \
            f"H_true wrong size at n={n}."

        # ----------------------------------------------------
        # Build all M deformation-dependent channels and TE estimates
        # ----------------------------------------------------
        for m in range(M):

            Hm = build_H_fim_from_paths(
                params,
                P_B_all[m], Zeta_B_all[m],
                P_U_all[m], Zeta_U_all[m],
                wavelength, fs, K
            )

            assert Hm.shape == (N_B, N_U, K), \
                f"Hm wrong size at n={n}, m={m}."

            Ym, sigma2, _ = pilot_transmission_multi(
                Hm, N_B, N_U, S_all, SNR_use, K
            )

            Rm = np.zeros((N_B, N_U, K), dtype=complex)

            for kk in range(K):
                # Ym[:, :, kk, :] has shape (N_B, N_U, Tpilots)
                Yk_all = Ym[:, :, kk, :]

                # In your MATLAB code, lambda_r is overwritten by sigma2
                lambda_use = sigma2

                Rm[:, :, kk] = ridge_TE_multipilot(
                    Yk_all,
                    S_all,
                    lambda_use
                )

            assert Rm.shape == (N_B, N_U, K), \
                f"Rm wrong size at n={n}, m={m}."

            H_all.append(Hm)
            R_all.append(Rm)

        # ----------------------------------------------------
        # One sample per adjacent subcarrier pair
        # ----------------------------------------------------
        for k in range(K - 1):

            # ------------------------------------------------
            # True undeformed target
            # ------------------------------------------------
            H0 = H_true[:, :, k]
            H1 = H_true[:, :, k + 1]

            R0_ref_raw = R_all[0][:, :, k]
            R1_ref_raw = R_all[0][:, :, k + 1]

            # ------------------------------------------------
            # Common normalization across all M views and target
            # ------------------------------------------------
            tmp_list = []

            for m in range(M):
                tmp_list.append(R_all[m][:, :, k].reshape(-1))
                tmp_list.append(R_all[m][:, :, k + 1].reshape(-1))

            tmp_list.append(H0.reshape(-1))
            tmp_list.append(H1.reshape(-1))

            tmp = np.concatenate(tmp_list)

            scale_c = np.max(np.abs(tmp)) + 1e-8

            # ------------------------------------------------
            # CNN input: stacked normalized R views
            # ------------------------------------------------
            Xin = np.zeros((N_B, N_U, 4 * M), dtype=np.float32)

            for m in range(M):
                R0m = R_all[m][:, :, k] / scale_c
                R1m = R_all[m][:, :, k + 1] / scale_c

                ch0 = 4 * m

                Xin[:, :, ch0 + 0] = np.real(R0m).astype(np.float32)
                Xin[:, :, ch0 + 1] = np.imag(R0m).astype(np.float32)
                Xin[:, :, ch0 + 2] = np.real(R1m).astype(np.float32)
                Xin[:, :, ch0 + 3] = np.imag(R1m).astype(np.float32)

            # ------------------------------------------------
            # Residual target:
            # Delta = H_ref_norm - R_ref_norm
            # ------------------------------------------------
            H0n = H0 / scale_c
            H1n = H1 / scale_c

            R0n_ref = R0_ref_raw / scale_c
            R1n_ref = R1_ref_raw / scale_c

            Delta0 = H0n - R0n_ref
            Delta1 = H1n - R1n_ref

            X[:, :, :, idx] = Xin

            V[:, :, 0, idx] = np.real(Delta0).astype(np.float32)
            V[:, :, 1, idx] = np.imag(Delta0).astype(np.float32)
            V[:, :, 2, idx] = np.real(Delta1).astype(np.float32)
            V[:, :, 3, idx] = np.imag(Delta1).astype(np.float32)

            idx += 1

    return X, V
