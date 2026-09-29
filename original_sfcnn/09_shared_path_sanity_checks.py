"""Exported verbatim from notebook code cell 9."""

#sanity check
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SANITY CHECK FOR build_H_fim_from_paths()
# Requires:
#   generateFIMsystem()
#   steering_vector_fim()
#   build_H_fim_from_paths()
# ============================================================

np.random.seed(1)

# ------------------------------------------------------------
# Physical parameters
# ------------------------------------------------------------
c = 3e8
fc = 28e9
wavelength = c / fc

fs = 100e3
K = 32
L = 3

# ------------------------------------------------------------
# FIM dimensions
# ------------------------------------------------------------
NH_B, NV_B = 5, 5
NH_U, NV_U = 5, 5

N_B = NH_B * NV_B
N_U = NH_U * NV_U

dxB = wavelength / 8
dyB = wavelength / 8
dxU = wavelength / 8
dyU = wavelength / 8

# ------------------------------------------------------------
# Generate FIM geometry
# ------------------------------------------------------------
P_B0, Zeta_B, P_U0, Zeta_U = generateFIMsystem(
    NH_B, NV_B, dxB, dyB,
    NH_U, NV_U, dxU, dyU,
    fc,
    flag=False
)

# ------------------------------------------------------------
# Generate path parameters
# Same format expected by build_H_fim_from_paths()
# ------------------------------------------------------------
params = {
    "AOA_az": (np.random.rand(L) - 0.5) * 2 * np.pi,
    "AOA_el": (np.random.rand(L) - 0.5) * np.pi,

    "DOA_az": (np.random.rand(L) - 0.5) * 2 * np.pi,
    "DOA_el": (np.random.rand(L) - 0.5) * np.pi,

    "BETA": (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2 * L),

    "delay": np.random.rand(L) * (1 / fs)
}

# ------------------------------------------------------------
# Build OFDM channel
# ------------------------------------------------------------
H = build_H_fim_from_paths(
    params,
    P_B0, Zeta_B,
    P_U0, Zeta_U,
    wavelength,
    fs,
    K
)

# ------------------------------------------------------------
# Print dimensions
# ------------------------------------------------------------
print("===================================================")
print("build_H_fim_from_paths sanity check")
print("===================================================")

print("P_B0 shape   :", P_B0.shape)
print("Zeta_B shape :", Zeta_B.shape)
print("P_U0 shape   :", P_U0.shape)
print("Zeta_U shape :", Zeta_U.shape)
print("H shape      :", H.shape)

# ------------------------------------------------------------
# Assertions
# ------------------------------------------------------------
assert P_B0.shape == (3, N_B)
assert Zeta_B.shape == (3, N_B)
assert P_U0.shape == (3, N_U)
assert Zeta_U.shape == (3, N_U)

assert H.shape == (N_B, N_U, K)
assert np.iscomplexobj(H)
assert np.all(np.isfinite(H))

# ------------------------------------------------------------
# Energy checks
# ------------------------------------------------------------
H_energy = np.mean(np.abs(H) ** 2)
H_norm = np.linalg.norm(H)

print("\nMean |H|^2 :", H_energy)
print("||H||_F    :", H_norm)

assert H_energy > 0
assert H_norm > 0

# ------------------------------------------------------------
# Check rank of one subcarrier
# The rank should be <= L approximately, because H is sum of L paths
# ------------------------------------------------------------
k0 = 0
rank_H0 = np.linalg.matrix_rank(H[:, :, k0], tol=1e-10)

print(f"\nRank of H[:,:,{k0}] :", rank_H0)
print("Expected rank <= L  :", L)

assert rank_H0 <= L

# ------------------------------------------------------------
# Check that subcarriers are not all identical when delay exists
# ------------------------------------------------------------
diff_01 = np.linalg.norm(H[:, :, 1] - H[:, :, 0])

print("\n||H[:,:,1] - H[:,:,0]||_F :", diff_01)

assert diff_01 > 0

# ------------------------------------------------------------
# Visualize first subcarrier magnitude
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.imshow(np.abs(H[:, :, 0]), aspect="auto")
plt.colorbar(label="|H|")
plt.title("Magnitude of H[:, :, 0]")
plt.xlabel("UE element index")
plt.ylabel("BS element index")
plt.tight_layout()
plt.show()

print("\nSanity check passed successfully.")

