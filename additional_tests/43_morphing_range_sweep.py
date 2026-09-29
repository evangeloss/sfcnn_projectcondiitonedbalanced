"""Exported verbatim from notebook code cell 43."""

import numpy as np
import torch
import matplotlib.pyplot as plt
# ============================================================
# NMSE vs MORPHING RANGE b/lambda
# Train once at b = lambda, test for different b
# ============================================================

def scale_codebook_to_morphing_range(Zeta_B_all_base, Zeta_U_all_base, alpha, wavelength):
    """
    alpha = b/lambda

    This rescales each deformation so that its maximum element displacement
    is approximately b = alpha * lambda.
    """

    b = alpha * wavelength

    Zeta_B_scaled = []
    Zeta_U_scaled = []

    for ZB, ZU in zip(Zeta_B_all_base, Zeta_U_all_base):

        # BS side
        norm_B = np.linalg.norm(ZB, axis=0)
        max_B = np.max(norm_B) + 1e-12
        Zeta_B_scaled.append(ZB * (b / max_B))

        # UE side
        norm_U = np.linalg.norm(ZU, axis=0)
        max_U = np.max(norm_U) + 1e-12
        Zeta_U_scaled.append(ZU * (b / max_U))

    return Zeta_B_scaled, Zeta_U_scaled


def NMSE_CNN_vs_morphing_range(
    N_B, N_U, fc, fs, K,
    NchCNN,
    P_B_all, Zeta_B_all_base,
    P_U_all, Zeta_U_all_base,
    M, Tpilots,
    netSF,
    alpha_list,
    SNR_dB=10,
    L_list=[3, 4],
    device=None
):

    wavelength = 3e8 / fc

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netSF = netSF.to(device)
    netSF.eval()

    S_all = generate_multi_pilots(N_U, Tpilots)

    Zeta_B_zero = np.zeros((3, N_B), dtype=float)
    Zeta_U_zero = np.zeros((3, N_U), dtype=float)

    NMSE_alpha = np.zeros((len(L_list), len(alpha_list)))

    for ia, alpha in enumerate(alpha_list):

        print("\n=================================================")
        print(f"Evaluating morphing range b/lambda = {alpha:.2f}")
        print("=================================================")

        Zeta_B_all, Zeta_U_all = scale_codebook_to_morphing_range(
            Zeta_B_all_base,
            Zeta_U_all_base,
            alpha,
            wavelength
        )

        for iL, L in enumerate(L_list):

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

                    Ym, _, _ = pilot_transmission_multi(
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

            NMSE_alpha[iL, ia] = err_sum / (pow_sum + 1e-12)

            print(
                f"L = {L}, b/lambda = {alpha:.2f}, "
                f"NMSE = {NMSE_alpha[iL, ia]:.4e}, "
                f"NMSE(dB) = {10*np.log10(NMSE_alpha[iL, ia]):.2f} dB"
            )

    return NMSE_alpha


# ============================================================
# RUN EVALUATION
# ============================================================

alpha_list = np.arange(0.05, 1, 0.1)

SNR_eval_dB = 10
NchCNN = 100
L_list = [3, 4,5]
Tpilots = 1;

NMSE_morphing = NMSE_CNN_vs_morphing_range(
    N_B, N_U, fc, fs, K,
    NchCNN,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    M, Tpilots,
    netSF,
    alpha_list,
    SNR_dB=SNR_eval_dB,
    L_list=L_list,
    device=device
)

print("NMSE_morphing linear:")
print(NMSE_morphing)

print("NMSE_morphing dB:")
print(10 * np.log10(NMSE_morphing))


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(7.2, 5.2))

for iL, L in enumerate(L_list):
    plt.semilogy(
        alpha_list,
        NMSE_morphing[iL, :],
        "-o",
        linewidth=1.6,
        markersize=6,
        markerfacecolor="white",
        label=f"Proposed CNN, L={L}"
    )

plt.grid(True, which="both", linestyle="-", alpha=0.4)
plt.xlabel(r"Morphing range, $b/\lambda$", fontsize=11)
plt.ylabel("NMSE", fontsize=11)
plt.title(f"NMSE vs Morphing Range at SNR = {SNR_eval_dB} dB", fontsize=12)
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()
