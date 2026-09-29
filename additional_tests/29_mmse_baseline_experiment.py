"""Exported verbatim from notebook code cell 29."""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# ============================================================
# 1) SYSTEM PARAMETERS
# ============================================================

c = 3e8
fc = 28e9
wavelength = c / fc
fs = 100e3
K = 32

SNR_vec_dB = np.arange(-10, 25, 5)
L_list = [1, 2, 3, 4]

pilotPow = 1
flag = False

# N = 25
NH_B = 5
NV_B = 5
N_B = NH_B * NV_B

NH_U = 5
NV_U = 5
N_U = NH_U * NV_U

dxB = wavelength / 8
dyB = wavelength / 8
dxU = wavelength / 8
dyU = wavelength / 8

# Number of deformation views
M = 2

# Dataset sizes for MMSE covariance estimation
NchTrain_MMSE = 500
NchTest_MMSE = 100

saveFolder = "."
codebookFile = os.path.join(saveFolder, "deformation_codebook_MMSE_N25.pkl")

# ============================================================
# 2) CODEBOOK
# ============================================================

regenerateCodebook = True

if regenerateCodebook or not os.path.isfile(codebookFile):

    print("Generating deformation codebook...")

    P_B_all = []
    Zeta_B_all = []
    P_U_all = []
    Zeta_U_all = []

    for m in range(M):
        P_B, Zeta_B, P_U, Zeta_U = generateFIMsystem(
            NH_B, NV_B, dxB, dyB,
            NH_U, NV_U, dxU, dyU,
            fc,
            flag=flag
        )

        P_B_all.append(P_B)
        Zeta_B_all.append(Zeta_B)
        P_U_all.append(P_U)
        Zeta_U_all.append(Zeta_U)

    with open(codebookFile, "wb") as f:
        pickle.dump(
            {
                "P_B_all": P_B_all,
                "Zeta_B_all": Zeta_B_all,
                "P_U_all": P_U_all,
                "Zeta_U_all": Zeta_U_all,
                "M": M,
                "N_B": N_B,
                "N_U": N_U,
                "K": K,
                "fc": fc,
                "fs": fs
            },
            f
        )

else:

    print("Loading deformation codebook...")

    with open(codebookFile, "rb") as f:
        codebook_data = pickle.load(f)

    P_B_all = codebook_data["P_B_all"]
    Zeta_B_all = codebook_data["Zeta_B_all"]
    P_U_all = codebook_data["P_U_all"]
    Zeta_U_all = codebook_data["Zeta_U_all"]

print("Codebook ready.")

# ============================================================
# 3) ADAPTER: CHANNEL GENERATION
# ============================================================

def generate_one_channel_set(
    N_B, N_U, fc, fs, K, L,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    M
):
    """
    Generates:
        H0     : undeformed target channel, shape [N_B, N_U, K]
        H_defs : deformed channels, shape [N_B, N_U, K, M]

    IMPORTANT:
    This assumes you already have:
        generate_path_parameters(...)
        build_H_fim_from_paths(...)

    If your function names are slightly different, only change this block.
    """

    # Same path parameters for all deformations
    path_params = generate_path_parameters(L, fc, fs)

    # Undeformed target: use zero deformation
    Zeta_B_zero = np.zeros_like(Zeta_B_all[0])
    Zeta_U_zero = np.zeros_like(Zeta_U_all[0])

    H0 = build_H_fim_from_paths(
        path_params,
        P_B_all[0], Zeta_B_zero,
        P_U_all[0], Zeta_U_zero, wavelength, fs, K
    )
    

    H_defs = np.zeros((N_B, N_U, K, M), dtype=np.complex128)

    for m in range(M):
        H_defs[:, :, :, m] = build_H_fim_from_paths(
            path_params,
            P_B_all[m], Zeta_B_all[m],
            P_U_all[m], Zeta_U_all[m], wavelength, fs,K
        )

    return H0, H_defs


# ============================================================
# 4) PILOT / TENTATIVE ESTIMATE GENERATION
# ============================================================

def generate_R_from_H(H_defs, SNR_dB, pilotPow=1):
    """
    Input:
        H_defs : [N_B, N_U, K, M]

    Output:
        R_defs : [N_B, N_U, K, M]
    """

    N_B, N_U, K, M = H_defs.shape

    # Unitary DFT pilot
    S = np.fft.fft(np.eye(N_U)) / np.sqrt(N_U)

    R_defs = np.zeros_like(H_defs, dtype=np.complex128)

    snr_linear = 10 ** (SNR_dB / 10)

    for m in range(M):
        for k in range(K):

            H = H_defs[:, :, k, m]

            Y_clean = np.sqrt(pilotPow) * H @ S

            signal_power = np.mean(np.abs(Y_clean) ** 2)
            noise_power = signal_power / snr_linear

            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(*Y_clean.shape)
                + 1j * np.random.randn(*Y_clean.shape)
            )

            Y = Y_clean + noise

            # LS tentative estimate
            R = (1 / np.sqrt(pilotPow)) * Y @ S.conj().T

            R_defs[:, :, k, m] = R

    return R_defs


# ============================================================
# 5) BUILD MMSE DATASET
# ============================================================

def build_mmse_dataset(
    Nch, N_B, N_U, fc, fs, K, L, SNR_dB,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    M
):
    H0_data = np.zeros((N_B, N_U, K, Nch), dtype=np.complex128)
    R_data = np.zeros((N_B, N_U, K, M, Nch), dtype=np.complex128)

    for n in range(Nch):

        if (n + 1) % 50 == 0:
            print(f"  Generated {n + 1}/{Nch} channels")

        H0, H_defs = generate_one_channel_set(
            N_B, N_U, fc, fs, K, L,
            P_B_all, Zeta_B_all,
            P_U_all, Zeta_U_all,
            M
        )

        R_defs = generate_R_from_H(H_defs, SNR_dB, pilotPow=pilotPow)

        H0_data[:, :, :, n] = H0
        R_data[:, :, :, :, n] = R_defs

    return H0_data, R_data


# ============================================================
# 6) EMPIRICAL LMMSE / MMSE EVALUATION
# ============================================================

def nmse_mmse_evaluation(H0_train, R_train, H0_test, R_test, eps_factor=1e-6):

    N_B, N_U, K, Ntrain = H0_train.shape
    _, _, _, M, Ntest = R_test.shape

    Nh = N_B * N_U
    Nz = M * Nh

    C_hz = np.zeros((Nh, Nz), dtype=np.complex128)
    C_zz = np.zeros((Nz, Nz), dtype=np.complex128)

    num_samples = K * Ntrain

    for n in range(Ntrain):
        for k in range(K):

            h = H0_train[:, :, k, n].reshape(-1, 1)

            z = []
            for m in range(M):
                z.append(R_train[:, :, k, m, n].reshape(-1, 1))

            z = np.vstack(z)

            C_hz += h @ z.conj().T
            C_zz += z @ z.conj().T

    C_hz /= num_samples
    C_zz /= num_samples

    eps = eps_factor * np.trace(C_zz).real / Nz
    C_zz_reg = C_zz + eps * np.eye(Nz)

    W_mmse = C_hz @ np.linalg.pinv(C_zz_reg)

    err_power = 0.0
    sig_power = 0.0

    for n in range(Ntest):
        for k in range(K):

            h_true = H0_test[:, :, k, n].reshape(-1, 1)

            z = []
            for m in range(M):
                z.append(R_test[:, :, k, m, n].reshape(-1, 1))

            z = np.vstack(z)

            h_hat = W_mmse @ z

            err_power += np.linalg.norm(h_true - h_hat) ** 2
            sig_power += np.linalg.norm(h_true) ** 2

    nmse_linear = err_power / sig_power
    return nmse_linear


# ============================================================
# 7) MAIN MMSE EVALUATION LOOP
# ============================================================

NMSE_MMSE = np.zeros((len(L_list), len(SNR_vec_dB)))

for iL, L in enumerate(L_list):

    print("\n===================================================")
    print(f"MMSE evaluation for L = {L}")
    print("===================================================")

    for iSNR, SNR_dB in enumerate(SNR_vec_dB):

        print(f"\nSNR = {SNR_dB} dB")

        print("Building MMSE training covariance dataset...")
        H0_train, R_train = build_mmse_dataset(
            NchTrain_MMSE,
            N_B, N_U, fc, fs, K, L, SNR_dB,
            P_B_all, Zeta_B_all,
            P_U_all, Zeta_U_all,
            M
        )

        print("Building MMSE test dataset...")
        H0_test, R_test = build_mmse_dataset(
            NchTest_MMSE,
            N_B, N_U, fc, fs, K, L, SNR_dB,
            P_B_all, Zeta_B_all,
            P_U_all, Zeta_U_all,
            M
        )

        nmse_linear = nmse_mmse_evaluation(
            H0_train, R_train,
            H0_test, R_test
        )

        NMSE_MMSE[iL, iSNR] = nmse_linear

        print(
            f"L={L}, SNR={SNR_dB} dB, "
            f"MMSE NMSE={10*np.log10(nmse_linear):.4f} dB"
        )


# ============================================================
# 8) PLOT: MMSE ONLY
# ============================================================

plt.figure(figsize=(8, 6))

markers = ["-ko", "-ks", "-k<", "-kd"]

for iL, L in enumerate(L_list):

    plt.semilogy(
        SNR_vec_dB,
        NMSE_MMSE[iL, :],
        markers[iL],
        linewidth=1.5,
        markersize=7,
        markerfacecolor="white",
        label=f"MMSE, L={L}"
    )

plt.grid(True, which="both")
plt.xlabel("SNR (dB)")
plt.ylabel("NMSE")
plt.xlim([-10, 20])
plt.ylim([1e-4, 1e1])
plt.legend(loc="lower left")
plt.title("MMSE Benchmark for FIM Channel Estimation, N = 25")
plt.tight_layout()
plt.show()

# ============================================================
# 9) NUMERIC DISPLAY
# ============================================================

print("\nNMSE_MMSE linear =")
print(NMSE_MMSE)

print("\nNMSE_MMSE dB =")
print(10 * np.log10(NMSE_MMSE))
