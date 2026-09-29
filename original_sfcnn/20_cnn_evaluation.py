"""Exported verbatim from notebook code cell 20."""

def NMSE_CNNevaluation2(
    N_B, N_U, fc, fs, K, NchCNN,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    M, netSF,
    device=None
):
    """
    CNN NMSE evaluation, shared-path version.

    Assumes PyTorch model input:
        [batch, channels, height, width]

    Original MATLAB input:
        [N_B, N_U, 4*M, 1]
    Converted PyTorch input:
        [1, 4*M, N_B, N_U]
    """

    c0 = 3e8
    wavelength = c0 / fc

    SNR_vec_dB = np.arange(-10, 25, 5)
    L_list = [1, 2, 3, 4]

    NMSE_CNN = np.zeros((len(L_list), len(SNR_vec_dB)))
    inference_time_list = []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netSF = netSF.to(device)
    netSF.eval()

    for iL, L in enumerate(L_list):

        for iS, SNR_dB in enumerate(SNR_vec_dB):

            err_sum = 0.0
            pow_sum = 0.0

            for n in range(NchCNN):

                # ----------------------------------------------------
                # 1) Shared propagation environment
                # ----------------------------------------------------
                params = generate_path_parameters(L, fc, fs)
                R_all = []

                # ----------------------------------------------------
                # 1b) True undeformed channel target
                # ----------------------------------------------------
                Zeta_B_zero = np.zeros_like(Zeta_B_all[0])
                Zeta_U_zero = np.zeros_like(Zeta_U_all[0])

                H_true = build_H_fim_from_paths(
                    params,
                    P_B_all[0], Zeta_B_zero,
                    P_U_all[0], Zeta_U_zero,
                    wavelength, fs, K
                )

                # ----------------------------------------------------
                # 2) Build M deformation views of SAME channel
                # ----------------------------------------------------
                for m in range(M):

                    Hm = build_H_fim_from_paths(
                        params,
                        P_B_all[m], Zeta_B_all[m],
                        P_U_all[m], Zeta_U_all[m],
                        wavelength, fs, K
                    )

                    Tpilots = 1
                    S_all = generate_multi_pilots(N_U, Tpilots)

                    Ym, sigma2, _ = pilot_transmission_multi(
                        Hm,
                        N_B,
                        N_U,
                        S_all,
                        SNR_dB,
                        K
                    )

                    Rm = np.zeros((N_B, N_U, K), dtype=complex)

                    for kk in range(K):
                        Yk_all = Ym[:, :, kk, :]

                        # Same as MATLAB:
                        # lambda_ridge = sigma2
                        Rm[:, :, kk] = ridge_TE_multipilot(
                            Yk_all,
                            S_all,
                            lambda_r=sigma2
                        )

                    R_all.append(Rm)

                # ----------------------------------------------------
                # 3) Evaluate adjacent subcarrier pairs
                # ----------------------------------------------------
                for k in range(K - 1):

                    H0 = H_true[:, :, k]
                    H1 = H_true[:, :, k + 1]

                    # ------------------------------------------------
                    # Same normalization rule as MATLAB evaluation
                    # ------------------------------------------------
                    tmp_list = []

                    for m in range(M):
                        tmp_list.append(R_all[m][:, :, k].reshape(-1))
                        tmp_list.append(R_all[m][:, :, k + 1].reshape(-1))

                    tmp = np.concatenate(tmp_list)
                    c_scale = np.max(np.abs(tmp)) + 1e-8

                    # ------------------------------------------------
                    # Build CNN input [N_B, N_U, 4M]
                    # ------------------------------------------------
                    Xin = np.zeros((N_B, N_U, 4 * M), dtype=np.float32)

                    for m in range(M):
                        R0m = R_all[m][:, :, k] / c_scale
                        R1m = R_all[m][:, :, k + 1] / c_scale

                        ch = 4 * m

                        Xin[:, :, ch + 0] = np.real(R0m).astype(np.float32)
                        Xin[:, :, ch + 1] = np.imag(R0m).astype(np.float32)
                        Xin[:, :, ch + 2] = np.real(R1m).astype(np.float32)
                        Xin[:, :, ch + 3] = np.imag(R1m).astype(np.float32)

                    # ------------------------------------------------
                    # Reference TE estimate, normalized
                    # ------------------------------------------------
                    R0_ref = R_all[0][:, :, k] / c_scale
                    R1_ref = R_all[0][:, :, k + 1] / c_scale

                    # ------------------------------------------------
                    # CNN prediction
                    # PyTorch expects [batch, channels, height, width]
                    # ------------------------------------------------
                    Xin_torch = torch.tensor(
                        np.transpose(Xin, (2, 0, 1))[None, :, :, :],
                        dtype=torch.float32,
                        device=device
                    )

                    with torch.no_grad():

                        sync_if_cuda(device)
                        t0 = time.perf_counter()
                    
                        Ypred = netSF(Xin_torch)
                    
                        sync_if_cuda(device)
                        t1 = time.perf_counter()
                    
                    inference_time_list.append(t1 - t0)

                    # Back to numpy:
                    # [1, 4, N_B, N_U] -> [N_B, N_U, 4]
                    Ypred = (
                        Ypred
                        .detach()
                        .cpu()
                        .numpy()[0]
                    )

                    Ypred = np.transpose(Ypred, (1, 2, 0))

                    Delta0_hat = Ypred[:, :, 0] + 1j * Ypred[:, :, 1]
                    Delta1_hat = Ypred[:, :, 2] + 1j * Ypred[:, :, 3]

                    # ------------------------------------------------
                    # Residual reconstruction
                    # ------------------------------------------------
                    H0hat = c_scale * (R0_ref + Delta0_hat)
                    H1hat = c_scale * (R1_ref + Delta1_hat)

                    # ------------------------------------------------
                    # NMSE accumulation
                    # ------------------------------------------------
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
                f"L={L}, "
                f"SNR={SNR_dB} dB, "
                f"CNN NMSE={NMSE_CNN[iL, iS]:.4e}"
            )
            avg_inference_time = np.mean(inference_time_list)
            std_inference_time = np.std(inference_time_list)
            
            print("\n================ CNN INFERENCE TIME RESULTS ================")
            print(f"Average inference time per subcarrier-pair estimate: {avg_inference_time:.6e} s")
            print(f"Average inference time per subcarrier-pair estimate: {avg_inference_time * 1e3:.6f} ms")
            print(f"Std inference time: {std_inference_time * 1e3:.6f} ms")

    return NMSE_CNN, avg_inference_time, std_inference_time
