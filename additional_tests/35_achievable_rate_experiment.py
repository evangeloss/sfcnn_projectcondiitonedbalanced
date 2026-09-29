"""Exported verbatim from notebook code cell 35."""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

# ============================================================
# 1) SYSTEM PARAMETERS
# ============================================================

np.random.seed(1)
torch.manual_seed(1)

c = 3e8
fc = 28e9
wavelength = c / fc
fs = 100e3
K = 32

SNR_vec_dB = np.arange(-10, 25, 5)

L = 3
M = 2
Ns = 1          # one beam / one stream
Ntest = 100

pilotPow = 1
flag = False

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

saveFolder = "."
codebookFile = os.path.join(saveFolder, "deformation_codebook.pkl")
netFile = os.path.join(saveFolder, "SF_CNN_L3_pairs_NRNT.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 2) LOAD CODEBOOK
# ============================================================

with open(codebookFile, "rb") as f:
    codebook_data = pickle.load(f)

P_B_all = codebook_data["P_B_all"]
Zeta_B_all = codebook_data["Zeta_B_all"]
P_U_all = codebook_data["P_U_all"]
Zeta_U_all = codebook_data["Zeta_U_all"]

print("Codebook loaded.")

# ============================================================
# 3) LOAD TRAINED CNN
# ============================================================

netSF = SFCNN(M).to(device)

checkpoint = torch.load(netFile, map_location=device,weights_only = False)
netSF.load_state_dict(checkpoint["model_state_dict"])
netSF.eval()

print("Trained CNN loaded.")

# ============================================================
# 4) RATE FUNCTION
# ============================================================

def achievable_rate_from_estimate(H_true, H_hat, SNR_dB, Ns=1):
    """
    H_true : [N_B, N_U, K]
    H_hat  : [N_B, N_U, K]
    """

    N_B, N_U, K = H_true.shape
    rho = 10 ** (SNR_dB / 10)

    rate_sum = 0.0

    for k in range(K):

        Hk = H_true[:, :, k]
        Hhat_k = H_hat[:, :, k]

        # SVD precoder from estimated channel
        _, _, Vh = np.linalg.svd(Hhat_k, full_matrices=False)

        F = Vh.conj().T[:, :Ns]

        # Power normalization
        F = F / np.linalg.norm(F, "fro") * np.sqrt(Ns)

        A = np.eye(N_B) + (rho / Ns) * Hk @ F @ F.conj().T @ Hk.conj().T

        rate_k = np.real(np.log2(np.linalg.det(A)))

        rate_sum += rate_k

    return rate_sum / K


# ============================================================
# 5) CHANNEL + CNN ESTIMATION FUNCTION
# ============================================================

def generate_true_and_cnn_estimate(SNR_dB):
    """
    This function must return:

        H_true : [N_B, N_U, K]
        H_hat  : [N_B, N_U, K]

    It uses your existing channel generation and CNN input pipeline.
    """

    # --------------------------------------------------------
    # Generate one test sample using your existing dataset code
    # --------------------------------------------------------
    X, Y = generate_Dataset_multiDef_multipilot(
        N_B, N_U, fc, fs, L, K, 1, np.array([SNR_dB]),
        P_B_all, Zeta_B_all,
        P_U_all, Zeta_U_all,
        M,
        Tpilots=1
    )

    # X: [N_B, N_U, 4M, Nsamp]
    # Y: [N_B, N_U, 4,  Nsamp]

    X_sample = X[:, :, :, 0]
    Y_sample = Y[:, :, :, 0]

    # --------------------------------------------------------
    # True channel from label
    # Assumption:
    # Y channels are:
    # [Re(H_k), Im(H_k), Re(H_k+1), Im(H_k+1)]
    #
    # Since your dataset is pair-based, we approximate full K
    # by filling all subcarriers with the available target pair.
    # For exact OFDM rate, replace this with direct H0 generation.
    # --------------------------------------------------------

    H_true = np.zeros((N_B, N_U, K), dtype=np.complex128)

    H_true[:, :, 0] = Y_sample[:, :, 0] + 1j * Y_sample[:, :, 1]
    H_true[:, :, 1] = Y_sample[:, :, 2] + 1j * Y_sample[:, :, 3]

    for k in range(2, K):
        H_true[:, :, k] = H_true[:, :, 1]

    # --------------------------------------------------------
    # CNN prediction
    # --------------------------------------------------------

    X_torch = np.transpose(X_sample, (2, 0, 1))
    X_torch = torch.tensor(X_torch[None, :, :, :], dtype=torch.float32).to(device)

    with torch.no_grad():
        Y_pred = netSF(X_torch).cpu().numpy()[0]

    # Y_pred: [4, N_B, N_U]
    Y_pred = np.transpose(Y_pred, (1, 2, 0))

    H_hat = np.zeros((N_B, N_U, K), dtype=np.complex128)

    H_hat[:, :, 0] = Y_pred[:, :, 0] + 1j * Y_pred[:, :, 1]
    H_hat[:, :, 1] = Y_pred[:, :, 2] + 1j * Y_pred[:, :, 3]

    for k in range(2, K):
        H_hat[:, :, k] = H_hat[:, :, 1]

    return H_true, H_hat


# ============================================================
# 6) MAIN ACHIEVABLE RATE EVALUATION
# ============================================================

Rate_CNN = np.zeros(len(SNR_vec_dB))
Rate_Perfect = np.zeros(len(SNR_vec_dB))

for iSNR, SNR_dB in enumerate(SNR_vec_dB):

    print("\n===================================================")
    print(f"Evaluating achievable rate at SNR = {SNR_dB} dB")
    print("===================================================")

    rate_cnn_sum = 0.0
    rate_perf_sum = 0.0

    for n in range(Ntest):

        if (n + 1) % 20 == 0:
            print(f"  Test channel {n + 1}/{Ntest}")

        H_true, H_hat = generate_true_and_cnn_estimate(SNR_dB)

        rate_cnn_sum += achievable_rate_from_estimate(
            H_true,
            H_hat,
            SNR_dB,
            Ns=Ns
        )

        rate_perf_sum += achievable_rate_from_estimate(
            H_true,
            H_true,
            SNR_dB,
            Ns=Ns
        )

    Rate_CNN[iSNR] = rate_cnn_sum / Ntest
    Rate_Perfect[iSNR] = rate_perf_sum / Ntest

    print(f"Proposed CNN rate: {Rate_CNN[iSNR]:.4f} bits/s/Hz")
    print(f"Perfect CSI rate : {Rate_Perfect[iSNR]:.4f} bits/s/Hz")


# ============================================================
# 7) PLOT
# ============================================================

plt.figure(figsize=(7.2, 5.2))

plt.plot(
    SNR_vec_dB,
    Rate_CNN,
    "-s",
    linewidth=1.6,
    markersize=7,
    markerfacecolor="white",
    label="Proposed SF-CNN"
)

plt.plot(
    SNR_vec_dB,
    Rate_Perfect,
    "-o",
    linewidth=1.6,
    markersize=7,
    markerfacecolor="white",
    label="Perfect CSI"
)

plt.grid(True, linestyle="-", alpha=0.4)
plt.xlabel("SNR (dB)")
plt.ylabel("Achievable Rate (bits/s/Hz)")
plt.title("Achievable Rate vs SNR, N = 25, L = 3")
plt.legend()
plt.tight_layout()
plt.show()

print("\nRate_CNN =")
print(Rate_CNN)

print("\nRate_Perfect =")
print(Rate_Perfect)
