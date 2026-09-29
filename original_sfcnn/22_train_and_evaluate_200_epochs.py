"""Exported verbatim from notebook code cell 22."""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# MAIN SCRIPT: SF-CNN FIM MULTI-DEFORMATION TRAINING/EVALUATION
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

SNR_train_dB = 10
L_train = 3
SNR_set = np.array([0, 5, 10, 15, 20])

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

# Element spacing
dxB = wavelength / 8
dyB = wavelength / 8
dxU = wavelength / 8
dyU = wavelength / 8

# Number of deformation views
M = 1
# ============================================================
# 2) RUN MODES
# ============================================================

doTrain = True
regenerateCodebook = True

saveFolder = "."
codebookFile = os.path.join(saveFolder, "deformation_codebook.pkl")
netFile = os.path.join(saveFolder, "SF_CNN_L3_pairs_NRNT.pt")

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
        "K": K
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

# ------------------------------
# Codebook consistency checks
# ------------------------------
for m in range(M):

    assert P_B_all[m].shape == (3, N_B), \
        f"P_B_all[{m}] must be 3 x {N_B}"

    assert Zeta_B_all[m].shape == (3, N_B), \
        f"Zeta_B_all[{m}] must be 3 x {N_B}"

    assert P_U_all[m].shape == (3, N_U), \
        f"P_U_all[{m}] must be 3 x {N_U}"

    assert Zeta_U_all[m].shape == (3, N_U), \
        f"Zeta_U_all[{m}] must be 3 x {N_U}"

    assert P_B_all[m].shape == Zeta_B_all[m].shape, \
        f"BS p0/zeta size mismatch at m={m}"

    assert P_U_all[m].shape == Zeta_U_all[m].shape, \
        f"UE p0/zeta size mismatch at m={m}"

print("PASS: codebook dimensions are consistent.")

# ============================================================
# 4) TRAINING / LOADING NETWORK
# ============================================================

print("\n=========================================================")
print("STEP 2: NETWORK")
print("=========================================================")

if doTrain:

    NchTrain = 1000
    NchVal = 200

    print("Building TRAIN dataset...")

    Xtr, Ytr = generate_Dataset_multiDef_multipilot(
        N_B, N_U, fc, fs, L_train, K, NchTrain, SNR_set,
        P_B_all, Zeta_B_all,
        P_U_all, Zeta_U_all,
        M,
        Tpilots=1
        
    )

    print("Building VAL dataset...")

    Xva, Yva = generate_Dataset_multiDef_multipilot(
        N_B, N_U, fc, fs, L_train, K, NchVal, SNR_set,
        P_B_all, Zeta_B_all,
        P_U_all, Zeta_U_all,
        M,
        Tpilots=1
    )

    print("Training data size:", Xtr.shape)
    print("Validation data size:", Xva.shape)

    # MATLAB:
    # Xtr : [N_B, N_U, 4M, Ns]
    # Ytr : [N_B, N_U, 4, Ns]
    #
    # PyTorch:
    # Xtr : [Ns, 4M, N_B, N_U]
    # Ytr : [Ns, 4,  N_B, N_U]

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
    netSF = SFCNN(M).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(netSF.parameters(), lr=1e-3)

    max_epochs = 50
    gradient_clip = 1.0

    print("*** Training SF-CNN ***")

    train_loss_history = []
    val_loss_history = []
    epoch_time_history = []
    
    total_training_start = time.perf_counter()
    
    for epoch in range(max_epochs):
    
        netSF.train()
        train_loss_sum = 0.0
    
        sync_if_cuda(device)
        epoch_start = time.perf_counter()
    
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
    
        sync_if_cuda(device)
        epoch_end = time.perf_counter()
    
        epoch_time = epoch_end - epoch_start
        epoch_time_history.append(epoch_time)
    
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
            f"Val Loss: {val_loss:.4e} "
            f"Time: {epoch_time:.3f} s"
        )
    
    sync_if_cuda(device)
    total_training_end = time.perf_counter()
    
    total_training_time = total_training_end - total_training_start
    avg_epoch_time = np.mean(epoch_time_history)
    train_time_per_channel_realisation_per_epoch = avg_epoch_time / len(train_loader.dataset)
    
    print("\n================ TRAINING TIME RESULTS ================")
    print(f"Total training time: {total_training_time:.3f} s")
    print(f"Average training time per epoch: {avg_epoch_time:.3f} s")
    print(
        f"Training time per channel realisation per epoch: "
        f"{train_time_per_channel_realisation_per_epoch:.6e} s"
    )
    # --------------------------------------------------------
    # Save trained network
    # --------------------------------------------------------
    checkpoint = {
        "model_state_dict": netSF.state_dict(),
        "L_train": L_train,
        "SNR_train_dB": SNR_train_dB,
        "M": M,
        "N_B": N_B,
        "N_U": N_U,
        "K": K,
        "fc": fc,
        "fs": fs,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "epoch_time_history": epoch_time_history,
        "total_training_time": total_training_time,
        "avg_epoch_time": avg_epoch_time,
        "train_time_per_channel_realisation_per_epoch": train_time_per_channel_realisation_per_epoch
    }

    torch.save(checkpoint, netFile)

    print(f"Trained network saved to: {netFile}")

else:

    print("Loading trained SF-CNN...")

    netSF = SFCNN(M).to(device)

    checkpoint = torch.load(netFile, map_location=device)

    netSF.load_state_dict(checkpoint["model_state_dict"])

# ============================================================
# 5) EVALUATION
# ============================================================

print("\n=========================================================")
print("STEP 3: EVALUATION")
print("=========================================================")

NchLS = 50
NchCNN = 50

print("Evaluating LS baseline...")

NMSE_LS = NMSE_LSevaluation(
    N_B, N_U, fc, fs, K, NchLS,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    pilotPow=pilotPow
)

print("Evaluating CNN...")

NMSE_CNN, avg_inf_time, std_inf_time = NMSE_CNNevaluation2(
    N_B, N_U, fc, fs, K, NchCNN,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    M,
    netSF,
    device=device
)

# ============================================================
# 6) PLOT
# ============================================================

print("\n=========================================================")
print("STEP 4: PLOT")
print("=========================================================")

SNR_vec_dB = np.arange(-10, 25, 5)
L_list = [1, 2, 3, 4]

plt.figure(figsize=(8, 6))

# LS, L=3 row index 2
plt.semilogy(
    SNR_vec_dB,
    NMSE_LS[2, :],
    "-k*",
    linewidth=1.5,
    markersize=7,
    label="LS (L=3)"
)

markers = ["-ko", "-ks", "-k<", "-kd"]

for iL, L in enumerate(L_list):

    plt.semilogy(
        SNR_vec_dB,
        NMSE_CNN[iL, :],
        markers[iL],
        linewidth=1.5,
        markersize=7,
        markerfacecolor="white",
        label=f"SF-CNN, L={L}" + (" (training)" if L == L_train else "")
    )

plt.grid(True, which="both")
plt.xlabel("SNR (dB)")
plt.ylabel("NMSE")
plt.xlim([-10, 20])
plt.ylim([1e-4, 1e1])
plt.legend(loc="lower left")
plt.title("Robustness of SF-CNN to different numbers of paths")
plt.tight_layout()
plt.show()

# ============================================================
# 7) OPTIONAL NUMERIC DISPLAY
# ============================================================

print("\nNMSE_CNN =")
print(NMSE_CNN)

print("\nNMSE_LS =")
print(NMSE_LS)

full_channel_inf_time = avg_inf_time * (K - 1)

print(f"Estimated full-channel inference time: {full_channel_inf_time * 1e3:.6f} ms")
