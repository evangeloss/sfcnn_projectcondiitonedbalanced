"""Exported verbatim from notebook code cell 11."""

# ============================================================
# SANITY TEST FOR ALL FIM FUNCTIONS
# ============================================================

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
NH_B = 5
NV_B = 5

NH_U = 5
NV_U = 5

dxB = wavelength / 8
dyB = wavelength / 8

dxU = wavelength / 8
dyU = wavelength / 8

# ============================================================
# 1) GENERATE FIM SYSTEM
# ============================================================

P_B_mat, Zeta_B, P_U_mat, Zeta_U = generateFIMsystem(
    NH_B, NV_B, dxB, dyB,
    NH_U, NV_U, dxU, dyU,
    fc,
    flag=False
)

print("===================================================")
print("FIM SYSTEM")
print("===================================================")

print("P_B_mat shape :", P_B_mat.shape)
print("Zeta_B shape  :", Zeta_B.shape)

print("P_U_mat shape :", P_U_mat.shape)
print("Zeta_U shape  :", Zeta_U.shape)

# ============================================================
# 2) TEST STEERING VECTOR
# ============================================================

phi = np.pi / 4
theta = np.pi / 3

a_B = steering_vector_fim(
    P_B_mat,
    Zeta_B,
    wavelength,
    phi,
    theta
)

print("\n===================================================")
print("STEERING VECTOR")
print("===================================================")

print("a_B shape :", a_B.shape)

norm_a = np.linalg.norm(a_B)
print("||a_B|| =", norm_a)

# Should be ~1
assert np.isclose(norm_a, 1.0, atol=1e-10)

# ============================================================
# 3) TEST CHANNEL GENERATION
# ============================================================

H, At, Ar, AOA, DOA, BETA, delay = generate_H_fim_deformed(
    P_B_mat,
    Zeta_B,
    P_U_mat,
    Zeta_U,
    wavelength,
    fs,
    K,
    L
)

print("\n===================================================")
print("CHANNEL GENERATION")
print("===================================================")

print("H shape     :", H.shape)
print("At shape    :", At.shape)
print("Ar shape    :", Ar.shape)

print("AOA shape   :", AOA.shape)
print("DOA shape   :", DOA.shape)

print("BETA shape  :", BETA.shape)
print("delay shape :", delay.shape)

# ============================================================
# 4) BASIC SANITY CHECKS
# ============================================================

assert H.shape == (NH_U * NV_U, NH_B * NV_B, K)
assert At.shape == (NH_B * NV_B, L)
assert Ar.shape == (NH_U * NV_U, L)

assert AOA.shape == (L, 2)
assert DOA.shape == (L, 2)

assert len(BETA) == L
assert len(delay) == L

print("\nAll dimension checks passed.")

# ============================================================
# 5) CHANNEL ENERGY CHECK
# ============================================================

channel_energy = np.mean(np.abs(H)**2)

print("\nAverage channel energy =", channel_energy)

assert channel_energy > 0

# ============================================================
# 6) PLOT SAMPLE SUBCARRIER MAGNITUDE
# ============================================================

k_plot = 0

plt.figure(figsize=(6, 5))

plt.imshow(
    np.abs(H[:, :, k_plot]),
    aspect='auto'
)

plt.colorbar(label='|H|')

plt.title(f'Channel Magnitude |H[:,:,{k_plot}]|')
plt.xlabel('BS elements')
plt.ylabel('UE elements')

plt.tight_layout()
plt.show()

# ============================================================
# 7) OPTIONAL GEOMETRY PLOT
# ============================================================

# Recover geometry in Nx3 format for plotting
P_B = P_B_mat.T
P_U = P_U_mat.T

P_B_def = P_B + Zeta_B.T
P_U_def = P_U + Zeta_U.T

plot_fim_system(
    P_B,
    P_B_def,
    P_U,
    P_U_def,
    R_B=None,
    R_U=None,
    NH_B=NH_B,
    NV_B=NV_B,
    NH_U=NH_U,
    NV_U=NV_U
)

print("\nSanity test completed successfully.")
