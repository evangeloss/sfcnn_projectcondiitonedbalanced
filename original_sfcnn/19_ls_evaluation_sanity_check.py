"""Exported verbatim from notebook code cell 19."""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SANITY CHECK FOR NMSE_LSevaluation()
# Requires:
#   generateFIMsystem
#   generate_path_parameters
#   steering_vector_fim
#   build_H_fim_from_paths
#   generate_multi_pilots
#   pilot_transmission_multi
#   ridge_TE_multipilot
#   NMSE_LSevaluation
# ============================================================

np.random.seed(1)

# ------------------------------------------------------------
# Physical parameters
# ------------------------------------------------------------
c = 3e8
fc = 28e9
wavelength = c / fc
fs = 100e3
K = 8

# ------------------------------------------------------------
# Small FIM dimensions for quick sanity test
# ------------------------------------------------------------
NH_B, NV_B = 4, 4
NH_U, NV_U = 4, 4

N_B = NH_B * NV_B
N_U = NH_U * NV_U

dxB = wavelength / 8
dyB = wavelength / 8
dxU = wavelength / 8
dyU = wavelength / 8

# ------------------------------------------------------------
# Generate one reference deformation view
# ------------------------------------------------------------
P_B0, Zeta_B, P_U0, Zeta_U = generateFIMsystem(
    NH_B, NV_B, dxB, dyB,
    NH_U, NV_U, dxU, dyU,
    fc,
    flag=False
)

P_B_all = [P_B0]
Zeta_B_all = [Zeta_B]
P_U_all = [P_U0]
Zeta_U_all = [Zeta_U]

# ------------------------------------------------------------
# Monte Carlo samples
# Keep small first
# ------------------------------------------------------------
NchLS = 5

# ------------------------------------------------------------
# Run LS evaluation
# ------------------------------------------------------------
NMSE_LS = NMSE_LSevaluation(
    N_B, N_U, fc, fs, K, NchLS,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    pilotPow=None
)

print("\n===================================================")
print("LS EVALUATION SANITY CHECK")
print("===================================================")

print("NMSE_LS shape:", NMSE_LS.shape)
print("NMSE_LS:")
print(NMSE_LS)

# ------------------------------------------------------------
# Expected shape
# L_list = [1,2,3,4]
# SNR_vec_dB = [-10,-5,0,5,10,15,20]
# ------------------------------------------------------------
expected_shape = (4, 7)

assert NMSE_LS.shape == expected_shape, "NMSE_LS has wrong shape."
assert np.all(np.isfinite(NMSE_LS)), "NMSE_LS contains NaN or Inf."
assert np.all(NMSE_LS > 0), "NMSE_LS should be positive."

print("\nShape and finite-value checks passed.")

# ------------------------------------------------------------
# Convert to dB
# ------------------------------------------------------------
NMSE_LS_dB = 10 * np.log10(NMSE_LS)

print("\nNMSE_LS_dB:")
print(NMSE_LS_dB)

# ------------------------------------------------------------
# Basic trend check:
# NMSE should generally decrease as SNR increases
# Because NchLS is small, allow small fluctuations.
# ------------------------------------------------------------
for i in range(NMSE_LS.shape[0]):
    start_val = NMSE_LS[i, 0]
    end_val = NMSE_LS[i, -1]

    print(
        f"L index {i}: "
        f"NMSE at -10 dB = {start_val:.4e}, "
        f"NMSE at 20 dB = {end_val:.4e}"
    )

    assert end_val < start_val, \
        "NMSE should be lower at high SNR than low SNR."

print("\nSNR trend check passed.")

# ------------------------------------------------------------
# Plot NMSE vs SNR
# ------------------------------------------------------------
SNR_vec_dB = np.arange(-10, 25, 5)
L_list = [1, 2, 3, 4]

plt.figure(figsize=(7, 5))

for i, L in enumerate(L_list):
    plt.semilogy(
        SNR_vec_dB,
        NMSE_LS[i, :],
        marker="o",
        label=f"L={L}"
    )

plt.grid(True, which="both")
plt.xlabel("SNR [dB]")
plt.ylabel("NMSE")
plt.title("LS / TE NMSE sanity check")
plt.legend()
plt.tight_layout()
plt.show()

print("\nLS evaluation sanity check completed successfully.")
