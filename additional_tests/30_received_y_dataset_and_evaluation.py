"""Exported verbatim from notebook code cell 30."""

def generate_Dataset_multiDef_receivedY(
    N_B, N_U, fc, fs, L, K, Nch, SNR_set,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    M, Tpilots=1
):
    """
    Dataset:
        Input  X = received pilot observations Y_m
        Label  Y = undeformed channel H_true

    No TE.
    No LS.
    No residual learning.
    """

    wavelength = 3e8 / fc
    numPairs = K - 1

    # Each deformation contributes:
    # Y(k) and Y(k+1), real/imag, for every pilot
    #
    # channels = 4 * M * Tpilots
    X = np.zeros(
        (N_B, N_U, 4 * M * Tpilots, Nch * numPairs),
        dtype=np.float32
    )

    Label = np.zeros(
        (N_B, N_U, 4, Nch * numPairs),
        dtype=np.float32
    )

    S_all = generate_multi_pilots(N_U, Tpilots)

    Zeta_B_zero = np.zeros((3, N_B), dtype=float)
    Zeta_U_zero = np.zeros((3, N_U), dtype=float)

    SNR_set = np.asarray(SNR_set).reshape(-1)

    idx = 0

    for n in range(Nch):

        if SNR_set.size == 1:
            SNR_use = float(SNR_set[0])
        else:
            SNR_use = float(np.random.choice(SNR_set))

        params = generate_path_parameters(L, fc, fs)

        # ----------------------------------------------------
        # Undeformed target channel H0
        # ----------------------------------------------------
        H_true = build_H_fim_from_paths(
            params,
            P_B_all[0], Zeta_B_zero,
            P_U_all[0], Zeta_U_zero,
            wavelength, fs, K
        )

        Y_all = []

        # ----------------------------------------------------
        # Received observations for every deformation
        # ----------------------------------------------------
        for m in range(M):

            Hm = build_H_fim_from_paths(
                params,
                P_B_all[m], Zeta_B_all[m],
                P_U_all[m], Zeta_U_all[m],
                wavelength, fs, K
            )

            Ym, sigma2, _ = pilot_transmission_multi(
                Hm,
                N_B,
                N_U,
                S_all,
                SNR_use,
                K
            )

            # Ym shape: [N_B, N_U, K, Tpilots]
            Y_all.append(Ym)

        # ----------------------------------------------------
        # Adjacent subcarrier samples
        # ----------------------------------------------------
        for k in range(K - 1):

            tmp_list = []

            for m in range(M):
                tmp_list.append(Y_all[m][:, :, k, :].reshape(-1))
                tmp_list.append(Y_all[m][:, :, k + 1, :].reshape(-1))

            tmp_list.append(H_true[:, :, k].reshape(-1))
            tmp_list.append(H_true[:, :, k + 1].reshape(-1))

            scale_c = np.max(np.abs(np.concatenate(tmp_list))) + 1e-8

            Xin = np.zeros(
                (N_B, N_U, 4 * M * Tpilots),
                dtype=np.float32
            )

            for m in range(M):

                for t in range(Tpilots):

                    Y0 = Y_all[m][:, :, k, t] / scale_c
                    Y1 = Y_all[m][:, :, k + 1, t] / scale_c

                    ch = 4 * (m * Tpilots + t)

                    Xin[:, :, ch + 0] = np.real(Y0).astype(np.float32)
                    Xin[:, :, ch + 1] = np.imag(Y0).astype(np.float32)
                    Xin[:, :, ch + 2] = np.real(Y1).astype(np.float32)
                    Xin[:, :, ch + 3] = np.imag(Y1).astype(np.float32)

            H0 = H_true[:, :, k] / scale_c
            H1 = H_true[:, :, k + 1] / scale_c

            X[:, :, :, idx] = Xin

            Label[:, :, 0, idx] = np.real(H0).astype(np.float32)
            Label[:, :, 1, idx] = np.imag(H0).astype(np.float32)
            Label[:, :, 2, idx] = np.real(H1).astype(np.float32)
            Label[:, :, 3, idx] = np.imag(H1).astype(np.float32)

            idx += 1

    return X, Label

def NMSE_CNNevaluation_receivedY(
    N_B, N_U, fc, fs, K, NchCNN,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    M, Tpilots,
    netSF,
    device=None
):

    wavelength = 3e8 / fc

    SNR_vec_dB = np.arange(-10, 25, 5)
    L_list = [1, 2, 3, 4]

    NMSE_CNN = np.zeros((len(L_list), len(SNR_vec_dB)))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netSF = netSF.to(device)
    netSF.eval()

    S_all = generate_multi_pilots(N_U, Tpilots)

    Zeta_B_zero = np.zeros((3, N_B), dtype=float)
    Zeta_U_zero = np.zeros((3, N_U), dtype=float)

    for iL, L in enumerate(L_list):

        for iS, SNR_dB in enumerate(SNR_vec_dB):

            err_sum = 0.0
            pow_sum = 0.0

            for n in range(NchCNN):

                params = generate_path_parameters(L, fc, fs)

                H_true = build_H_fim_from_paths(
                    params,
                    P_B_all[0], Zeta_B_zero,
                    P_U_all[0], Zeta_U_zero,
                    wavelength, fs, K
                )

                Y_all = []

                for m in range(M):

                    Hm = build_H_fim_from_paths(
                        params,
                        P_B_all[m], Zeta_B_all[m],
                        P_U_all[m], Zeta_U_all[m],
                        wavelength, fs, K
                    )

                    Ym, sigma2, _ = pilot_transmission_multi(
                        Hm,
                        N_B,
                        N_U,
                        S_all,
                        SNR_dB,
                        K
                    )

                    Y_all.append(Ym)

                for k in range(K - 1):

                    tmp_list = []

                    for m in range(M):
                        tmp_list.append(Y_all[m][:, :, k, :].reshape(-1))
                        tmp_list.append(Y_all[m][:, :, k + 1, :].reshape(-1))

                    tmp_list.append(H_true[:, :, k].reshape(-1))
                    tmp_list.append(H_true[:, :, k + 1].reshape(-1))

                    scale_c = np.max(np.abs(np.concatenate(tmp_list))) + 1e-8

                    Xin = np.zeros(
                        (N_B, N_U, 4 * M * Tpilots),
                        dtype=np.float32
                    )

                    for m in range(M):

                        for t in range(Tpilots):

                            Y0 = Y_all[m][:, :, k, t] / scale_c
                            Y1 = Y_all[m][:, :, k + 1, t] / scale_c

                            ch = 4 * (m * Tpilots + t)

                            Xin[:, :, ch + 0] = np.real(Y0).astype(np.float32)
                            Xin[:, :, ch + 1] = np.imag(Y0).astype(np.float32)
                            Xin[:, :, ch + 2] = np.real(Y1).astype(np.float32)
                            Xin[:, :, ch + 3] = np.imag(Y1).astype(np.float32)

                    Xin_torch = torch.tensor(
                        np.transpose(Xin, (2, 0, 1))[None, :, :, :],
                        dtype=torch.float32,
                        device=device
                    )

                    with torch.no_grad():
                        Ypred = netSF(Xin_torch)

                    Ypred = Ypred.detach().cpu().numpy()[0]
                    Ypred = np.transpose(Ypred, (1, 2, 0))

                    H0hat = scale_c * (Ypred[:, :, 0] + 1j * Ypred[:, :, 1])
                    H1hat = scale_c * (Ypred[:, :, 2] + 1j * Ypred[:, :, 3])

                    H0 = H_true[:, :, k]
                    H1 = H_true[:, :, k + 1]

                    err_sum += (
                        np.linalg.norm(H0hat - H0) ** 2
                        + np.linalg.norm(H1hat - H1) ** 2
                    )

                    pow_sum += (
                        np.linalg.norm(H0) ** 2
                        + np.linalg.norm(H1) ** 2
                    )

            NMSE_CNN[iL, iS] = err_sum / (pow_sum + 1e-12)

            print(
                f"L={L}, SNR={SNR_dB} dB, "
                f"CNN NMSE={NMSE_CNN[iL, iS]:.4e}"
            )

    return NMSE_CNN
