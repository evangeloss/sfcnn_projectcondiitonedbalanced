"""Exported verbatim from notebook code cell 38."""

# ============================================================
# MAIN: FIM-optimized deformation codebook
# ============================================================

c = 3e8
fc = 28e9
wavelength = c / fc
fs = 100e3
K = 32

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

L = 3
M = 8
Ncand = 100
SNR_dB = 10
flag = False

P_B_all, Zeta_B_all, P_U_all, Zeta_U_all, fim_history = optimize_fim_deformation_codebook(
    NH_B, NV_B, dxB, dyB,
    NH_U, NV_U, dxU, dyU,
    fc, fs, K,
    L,
    M,
    Ncand=Ncand,
    SNR_dB=SNR_dB,
    flag=flag,
    seed=1
)

codebookFile = "deformation_codebook_FIM_optimized.pkl"

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
            "fs": fs,
            "fim_history": fim_history
        },
        f
    )

print("Saved optimized FIM codebook to:", codebookFile)
