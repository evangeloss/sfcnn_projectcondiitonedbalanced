"""Exported verbatim from notebook code cell 39."""

# ============================================================
# RANDOM CODEBOOK BASELINE
# ============================================================

Nrandom = 20

random_logdet = []
random_traceCRB = []
random_condJ = []

sigma2 = 10**(-SNR_dB/10)

# Same channel realization used during optimization
params = generate_path_parameters(L, fc, fs)

S_all = generate_multi_pilots(N_U, Tpilots=1)
S = S_all[0]

for r in range(Nrandom):

    print(f"Random codebook {r+1}/{Nrandom}")

    P_B_rand_all = []
    Zeta_B_rand_all = []
    P_U_rand_all = []
    Zeta_U_rand_all = []

    for m in range(M):

        P_B, Zeta_B, P_U, Zeta_U = generateFIMsystem(
            NH_B, NV_B, dxB, dyB,
            NH_U, NV_U, dxU, dyU,
            fc,
            flag=False
        )

        P_B_rand_all.append(P_B)
        Zeta_B_rand_all.append(Zeta_B)
        P_U_rand_all.append(P_U)
        Zeta_U_rand_all.append(Zeta_U)

    J_rand, _ = compute_fim_for_codebook(
        params,
        P_B_rand_all, Zeta_B_rand_all,
        P_U_rand_all, Zeta_U_rand_all,
        S,
        fc, fs, K,
        sigma2
    )

    logdetJ, trCRB, condJ = score_fim_codebook(J_rand)

    random_logdet.append(logdetJ)
    random_traceCRB.append(trCRB)
    random_condJ.append(condJ)

# ============================================================
# RANDOM STATISTICS
# ============================================================

random_logdet = np.array(random_logdet)
random_traceCRB = np.array(random_traceCRB)

print("\n==============================")
print(" RANDOM CODEBOOK STATISTICS ")
print("==============================")

print("logdet(J)")
print("mean =", np.mean(random_logdet))
print("std  =", np.std(random_logdet))
print("best =", np.max(random_logdet))

print("\ntrace(CRB)")
print("mean =", np.mean(random_traceCRB))
print("std  =", np.std(random_traceCRB))
print("best =", np.min(random_traceCRB))
