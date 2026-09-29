"""Exported verbatim from notebook code cell 37."""

# ============================================================
# 5) Greedy FIM deformation codebook optimization
# ============================================================

def optimize_fim_deformation_codebook(
    NH_B, NV_B, dxB, dyB,
    NH_U, NV_U, dxU, dyU,
    fc, fs, K,
    L,
    M,
    Ncand=100,
    SNR_dB=10,
    flag=False,
    seed=1
):
    """
    Greedy selection of M FIM deformation states.

    At each step:
        1) generate Ncand random candidate deformations
        2) append each one to current codebook
        3) compute FIM
        4) keep candidate with largest logdet(J)

    Output:
        optimized P_B_all, Zeta_B_all, P_U_all, Zeta_U_all
    """

    np.random.seed(seed)

    N_B = NH_B * NV_B
    N_U = NH_U * NV_U

    sigma2 = 10 ** (-SNR_dB / 10)

    # Fixed propagation environment for codebook optimization
    params = generate_path_parameters(L, fc, fs)

    # Pilot matrix
    S_all = generate_multi_pilots(N_U, Tpilots=1)
    S = S_all[0]

    P_B_best_all = []
    Zeta_B_best_all = []
    P_U_best_all = []
    Zeta_U_best_all = []

    history = []

    for m in range(M):

        best_score = -np.inf
        best_candidate = None
        best_info = None

        print(f"\nSelecting deformation {m+1}/{M}")

        for c in range(Ncand):

            P_B, Zeta_B, P_U, Zeta_U = generateFIMsystem(
                NH_B, NV_B, dxB, dyB,
                NH_U, NV_U, dxU, dyU,
                fc,
                flag=flag
            )

            P_B_trial = P_B_best_all + [P_B]
            Zeta_B_trial = Zeta_B_best_all + [Zeta_B]
            P_U_trial = P_U_best_all + [P_U]
            Zeta_U_trial = Zeta_U_best_all + [Zeta_U]

            J, D = compute_fim_for_codebook(
                params,
                P_B_trial, Zeta_B_trial,
                P_U_trial, Zeta_U_trial,
                S,
                fc, fs, K,
                sigma2
            )

            logdetJ, trCRB, condJ = score_fim_codebook(J)

            if logdetJ > best_score:
                best_score = logdetJ
                best_candidate = (P_B, Zeta_B, P_U, Zeta_U)
                best_info = (logdetJ, trCRB, condJ)

        P_B_best_all.append(best_candidate[0])
        Zeta_B_best_all.append(best_candidate[1])
        P_U_best_all.append(best_candidate[2])
        Zeta_U_best_all.append(best_candidate[3])

        history.append({
            "m": m + 1,
            "logdetJ": best_info[0],
            "traceCRB": best_info[1],
            "condJ": best_info[2]
        })

        print("Best logdet(J):", best_info[0])
        print("trace(CRB):", best_info[1])
        print("cond(J):", best_info[2])

    return (
        P_B_best_all,
        Zeta_B_best_all,
        P_U_best_all,
        Zeta_U_best_all,
        history
    )
