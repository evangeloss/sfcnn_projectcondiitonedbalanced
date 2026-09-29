"""Exported verbatim from notebook code cell 23."""

def NMSE_CNN_vs_paths_avgSNR(
    N_B, N_U, fc, fs, K, NchCNN,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    M, netSF,
    SNR_vec_dB=np.array([-10, -5, 0, 5, 10, 15, 20]),
    L_list=np.arange(1, 11),
    Tpilots=1,
    device=None
):

    c0 = 3e8
    wavelength = c0 / fc

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netSF = netSF.to(device)
    netSF.eval()

    S_all = generate_multi_pilots(N_U, Tpilots)

    NMSE_CNN_avgSNR = np.zeros(len(L_list))
    NMSE_TE_avgSNR  = np.zeros(len(L_list))

    Zeta_B_zero = np.zeros_like(Zeta_B_all[0])
    Zeta_U_zero = np.zeros_like(Zeta_U_all[0])

    for iL, L in enumerate(L_list):

        print(f"Evaluating L = {L}")

        NMSE_CNN_perSNR = []
        NMSE_TE_perSNR  = []

        for SNR_dB in SNR_vec_dB:

            print(f"  SNR = {SNR_dB} dB")

            err_cnn_sum = 0.0
            err_te_sum  = 0.0
            pow_sum     = 0.0

            for n in range(NchCNN):

                params = generate_path_parameters(L, fc, fs)

                H_true = build_H_fim_from_paths(
                    params,
                    P_B_all[0], Zeta_B_zero,
                    P_U_all[0], Zeta_U_zero,
                    wavelength, fs, K
                )

                R_all = []

                for m in range(M):

                    Hm = build_H_fim_from_paths(
                        params,
                        P_B_all[m], Zeta_B_all[m],
                        P_U_all[m], Zeta_U_all[m],
                        wavelength, fs, K
                    )

                    Ym, sigma2, _ = pilot_transmission_multi(
                        Hm, N_B, N_U, S_all, SNR_dB, K
                    )

                    Rm = np.zeros((N_B, N_U, K), dtype=complex)

                    for kk in range(K):
                        Rm[:, :, kk] = ridge_TE_multipilot(
                            Ym[:, :, kk, :],
                            S_all,
                            lambda_r=sigma2
                        )

                    R_all.append(Rm)

                for k in range(K - 1):

                    H0 = H_true[:, :, k]
                    H1 = H_true[:, :, k + 1]

                    tmp_list = []

                    for m in range(M):
                        tmp_list.append(R_all[m][:, :, k].reshape(-1))
                        tmp_list.append(R_all[m][:, :, k + 1].reshape(-1))

                    tmp = np.concatenate(tmp_list)
                    c_scale = np.max(np.abs(tmp)) + 1e-8

                    Xin = np.zeros((N_B, N_U, 4 * M), dtype=np.float32)

                    for m in range(M):

                        R0m = R_all[m][:, :, k] / c_scale
                        R1m = R_all[m][:, :, k + 1] / c_scale

                        ch = 4 * m

                        Xin[:, :, ch + 0] = np.real(R0m).astype(np.float32)
                        Xin[:, :, ch + 1] = np.imag(R0m).astype(np.float32)
                        Xin[:, :, ch + 2] = np.real(R1m).astype(np.float32)
                        Xin[:, :, ch + 3] = np.imag(R1m).astype(np.float32)

                    Xin_torch = torch.tensor(
                        np.transpose(Xin, (2, 0, 1))[None, :, :, :],
                        dtype=torch.float32,
                        device=device
                    )

                    with torch.no_grad():
                        Ypred = netSF(Xin_torch)

                    Ypred = Ypred.detach().cpu().numpy()[0]
                    Ypred = np.transpose(Ypred, (1, 2, 0))

                    Delta0_hat = Ypred[:, :, 0] + 1j * Ypred[:, :, 1]
                    Delta1_hat = Ypred[:, :, 2] + 1j * Ypred[:, :, 3]

                    R0_ref = R_all[0][:, :, k] / c_scale
                    R1_ref = R_all[0][:, :, k + 1] / c_scale

                    H0hat_cnn = c_scale * (R0_ref + Delta0_hat)
                    H1hat_cnn = c_scale * (R1_ref + Delta1_hat)

                    H0hat_te = R_all[0][:, :, k]
                    H1hat_te = R_all[0][:, :, k + 1]

                    err_cnn_sum += (
                        np.linalg.norm(H0hat_cnn - H0) ** 2
                        + np.linalg.norm(H1hat_cnn - H1) ** 2
                    )

                    err_te_sum += (
                        np.linalg.norm(H0hat_te - H0) ** 2
                        + np.linalg.norm(H1hat_te - H1) ** 2
                    )

                    pow_sum += (
                        np.linalg.norm(H0) ** 2
                        + np.linalg.norm(H1) ** 2
                    )

            nmse_cnn_snr = err_cnn_sum / (pow_sum + 1e-12)
            nmse_te_snr  = err_te_sum  / (pow_sum + 1e-12)

            NMSE_CNN_perSNR.append(nmse_cnn_snr)
            NMSE_TE_perSNR.append(nmse_te_snr)

            print(
                f"    TE={10*np.log10(nmse_te_snr):.2f} dB, "
                f"CNN={10*np.log10(nmse_cnn_snr):.2f} dB"
            )

        NMSE_CNN_avgSNR[iL] = np.mean(NMSE_CNN_perSNR)
        NMSE_TE_avgSNR[iL]  = np.mean(NMSE_TE_perSNR)

        print(
            f"L={L}, AVG over SNRs: "
            f"TE={10*np.log10(NMSE_TE_avgSNR[iL]):.2f} dB, "
            f"CNN={10*np.log10(NMSE_CNN_avgSNR[iL]):.2f} dB"
        )

    return np.array(L_list), NMSE_TE_avgSNR, NMSE_CNN_avgSNR
