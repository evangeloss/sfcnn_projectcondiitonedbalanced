"""Exported verbatim from notebook code cell 31."""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# MAIN SCRIPT: Y(m) -> H0 DIRECT CNN TRAINING/EVALUATION
# ============================================================

np.random.seed(1)
torch.manual_seed(1)

# ============================================================
# 1) SYSTEM PARAMETERS
# ============================================================

c = 3e8
fc = 28e9
wavelength = c / fc
fs = 100e3
K = 32

SNR_train_set = np.array([0, 5, 10, 15, 20])
L_train = 3

pilotPow = 1
flag = False

# BS-FIM
NH_B = 5
NV_B = 5
N_B = NH_B * NV_B

# UE-FIM
NH_U = 5
NV_U = 5
N_U = NH_U * NV_U

dxB = wavelength / 8
dyB = wavelength / 8
dxU = wavelength / 8
dyU = wavelength / 8

# Number of deformation views
M = 8

# Number of pilots per deformation
Tpilots = 1

# IMPORTANT:
# CNN input channels = 4 * M * Tpilots
# label channels     = 4
CNN_input_factor = M * Tpilots


# ============================================================
# 2) RUN MODES
# ============================================================

doTrain = False
regenerateCodebook = True

saveFolder = "."
codebookFile = os.path.join(saveFolder, "deformation_codebook.pkl")
netFile = os.path.join(saveFolder, "SF_CNN_Y_to_H0_direct.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# 3) MULTI-DEFORMATION CODEBOOK
# ============================================================

print("\n=========================================================")
print("STEP 1: CODEBOOK")
print("=========================================================")

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

    codebook_data = {
        "P_B_all": P_B_all,
        "Zeta_B_all": Zeta_B_all,
        "P_U_all": P_U_all,
        "Zeta_U_all": Zeta_U_all,
        "M": M,
        "NH_B": NH_B,
        "NV_B": NV_B,
        "NH_U": NH_U,
        "NV_U": NV_U,
        "N_B": N_B,
        "N_U": N_U,
        "fc": fc,
        "fs": fs,
        "K": K,
        "Tpilots": Tpilots
    }

    with open(codebookFile, "wb") as f:
        pickle.dump(codebook_data, f)

    print(f"Codebook saved to: {codebookFile}")

else:

    print("Loading deformation codebook...")

    with open(codebookFile, "rb") as f:
        codebook_data = pickle.load(f)

    P_B_all = codebook_data["P_B_all"]
    Zeta_B_all = codebook_data["Zeta_B_all"]
    P_U_all = codebook_data["P_U_all"]
    Zeta_U_all = codebook_data["Zeta_U_all"]

    M = codebook_data["M"]
    NH_B = codebook_data["NH_B"]
    NV_B = codebook_data["NV_B"]
    NH_U = codebook_data["NH_U"]
    NV_U = codebook_data["NV_U"]
    N_B = codebook_data["N_B"]
    N_U = codebook_data["N_U"]


# ============================================================
# 4) CODEBOOK CHECKS
# ============================================================

for m in range(M):

    assert P_B_all[m].shape == (3, N_B)
    assert Zeta_B_all[m].shape == (3, N_B)

    assert P_U_all[m].shape == (3, N_U)
    assert Zeta_U_all[m].shape == (3, N_U)

print("PASS: codebook dimensions are consistent.")


# ============================================================
# 5) TRAINING / LOADING NETWORK
# ============================================================

print("\n=========================================================")
print("STEP 2: NETWORK")
print("=========================================================")

if doTrain:

    NchTrain = 1000
    NchVal = 200

    print("Building TRAIN dataset: Y(m) -> H0")

    Xtr, Ytr = generate_Dataset_multiDef_receivedY(
        N_B, N_U, fc, fs,
        L_train, K, NchTrain,
        SNR_train_set,
        P_B_all, Zeta_B_all,
        P_U_all, Zeta_U_all,
        M,
        Tpilots=Tpilots
    )

    print("Building VAL dataset: Y(m) -> H0")

    Xva, Yva = generate_Dataset_multiDef_receivedY(
        N_B, N_U, fc, fs,
        L_train, K, NchVal,
        SNR_train_set,
        P_B_all, Zeta_B_all,
        P_U_all, Zeta_U_all,
        M,
        Tpilots=Tpilots
    )

    print("Training data size:", Xtr.shape)
    print("Validation data size:", Xva.shape)

    # NumPy:
    # Xtr: [N_B, N_U, 4*M*Tpilots, Ns]
    # Ytr: [N_B, N_U, 4, Ns]
    #
    # PyTorch:
    # Xtr: [Ns, 4*M*Tpilots, N_B, N_U]
    # Ytr: [Ns, 4, N_B, N_U]

    Xtr_torch = np.transpose(Xtr, (3, 2, 0, 1))
    Ytr_torch = np.transpose(Ytr, (3, 2, 0, 1))

    Xva_torch = np.transpose(Xva, (3, 2, 0, 1))
    Yva_torch = np.transpose(Yva, (3, 2, 0, 1))

    Xtr_torch = torch.tensor(Xtr_torch, dtype=torch.float32)
    Ytr_torch = torch.tensor(Ytr_torch, dtype=torch.float32)

    Xva_torch = torch.tensor(Xva_torch, dtype=torch.float32)
    Yva_torch = torch.tensor(Yva_torch, dtype=torch.float32)

    train_dataset = TensorDataset(Xtr_torch, Ytr_torch)
    val_dataset = TensorDataset(Xva_torch, Yva_torch)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    # IMPORTANT:
    # SFCNN argument must produce input channels = 4 * CNN_input_factor
    # So if your SFCNN(M) uses input channels = 4*M,
    # then here we pass M*Tpilots.
    # --------------------------------------------------------

    netSF = SFCNN(CNN_input_factor).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(netSF.parameters(), lr=1e-3)

    max_epochs = 20
    gradient_clip = 1.0

    print("*** Training SF-CNN: direct Y(m) -> H0 ***")

    train_loss_history = []
    val_loss_history = []

    for epoch in range(max_epochs):

        netSF.train()
        train_loss_sum = 0.0

        for Xbatch, Ybatch in train_loader:

            Xbatch = Xbatch.to(device)
            Ybatch = Ybatch.to(device)

            optimizer.zero_grad()

            Ypred = netSF(Xbatch)

            loss = criterion(Ypred, Ybatch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                netSF.parameters(),
                gradient_clip
            )

            optimizer.step()

            train_loss_sum += loss.item() * Xbatch.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        netSF.eval()
        val_loss_sum = 0.0

        with torch.no_grad():

            for Xbatch, Ybatch in val_loader:

                Xbatch = Xbatch.to(device)
                Ybatch = Ybatch.to(device)

                Ypred = netSF(Xbatch)

                loss = criterion(Ypred, Ybatch)

                val_loss_sum += loss.item() * Xbatch.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{max_epochs}] "
            f"Train Loss: {train_loss:.4e} "
            f"Val Loss: {val_loss:.4e}"
        )

    checkpoint = {
        "model_state_dict": netSF.state_dict(),
        "L_train": L_train,
        "SNR_train_set": SNR_train_set,
        "M": M,
        "Tpilots": Tpilots,
        "CNN_input_factor": CNN_input_factor,
        "N_B": N_B,
        "N_U": N_U,
        "K": K,
        "fc": fc,
        "fs": fs,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history
    }

    torch.save(checkpoint, netFile)

    print(f"Trained network saved to: {netFile}")

else:

    print("Loading trained SF-CNN...")

    checkpoint = torch.load(netFile, map_location=device, weights_only  = False)

    CNN_input_factor = checkpoint["CNN_input_factor"]
    Tpilots = checkpoint["Tpilots"]

    netSF = SFCNN(CNN_input_factor).to(device)
    netSF.load_state_dict(checkpoint["model_state_dict"])


# ============================================================
# 6) CNN EVALUATION ONLY
# ============================================================

print("\n=========================================================")
print("STEP 3: CNN EVALUATION")
print("=========================================================")

NchCNN = 50

NMSE_CNN = NMSE_CNNevaluation_receivedY(
    N_B, N_U, fc, fs, K, NchCNN,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    M,
    Tpilots,
    netSF,
    device=device
)


# ============================================================
# 7) PLOT CNN ONLY
# ============================================================

print("\n=========================================================")
print("STEP 4: PLOT")
print("=========================================================")

SNR_vec_dB = np.arange(-10, 25, 5)
L_list = [1, 2, 3, 4]

plt.figure(figsize=(8, 6))

markers = ["-ko", "-ks", "-k<", "-kd"]

for iL, L in enumerate(L_list):

    plt.semilogy(
        SNR_vec_dB,
        NMSE_CNN[iL, :],
        markers[iL],
        linewidth=1.5,
        markersize=7,
        markerfacecolor="white",
        label=f"SF-CNN, L={L}" + (" training" if L == L_train else "")
    )

plt.grid(True, which="both")
plt.xlabel("SNR (dB)")
plt.ylabel("NMSE")
plt.xlim([-10, 20])
plt.ylim([1e-4, 1e1])
plt.legend(loc="lower left")
plt.title("Direct received-signal CNN estimation: Y(m) to undeformed H0")
plt.tight_layout()
plt.show()


# ============================================================
# 8) PRINT VALUES
# ============================================================

print("\nNMSE_CNN =")
print(NMSE_CNN)
