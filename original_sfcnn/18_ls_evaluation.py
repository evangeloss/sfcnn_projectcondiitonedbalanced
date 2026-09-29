"""Exported verbatim from notebook code cell 18."""

import numpy as np


def NMSE_LSevaluation(
    N_B, N_U, fc, fs, K, NchLS,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    pilotPow=None
):
    """
    LS / TE evaluation (shared-path consistent version).

    Policy
    ------
    - Same path realization per Monte Carlo run
    - Only deformation m = 0 used for LS

    Returns
    -------
    NMSE_LS : ndarray, shape (len(L_list), len(SNR_vec_dB))
    """

    c0 = 3e8
    wavelength = c0 / fc

    SNR_vec_dB = np.arange(-10, 25, 5)
    L_list = [1, 2, 3, 4]

    # --------------------------------------------------------
    # Reference deformation
    # --------------------------------------------------------
    P_B_ref = P_B_all[0]
    Zeta_B_ref = Zeta_B_all[0]

    P_U_ref = P_U_all[0]
    Zeta_U_ref = Zeta_U_all[0]

    NMSE_LS = np.zeros((len(L_list), len(SNR_vec_dB)))

    # ========================================================
    # Loop over number of paths
    # ========================================================
    for iL, L in enumerate(L_list):

        # ====================================================
        # Loop over SNR
        # ====================================================
        for iS, SNR_dB in enumerate(SNR_vec_dB):

            err_sum = 0.0
            pow_sum = 0.0

            # =================================================
            # Monte Carlo runs
            # =================================================
            for n in range(NchLS):

                # --------------------------------------------
                # Shared path generation
                # --------------------------------------------
                params = generate_path_parameters(
                    L, fc, fs
                )

                # --------------------------------------------
                # Build channel
                # --------------------------------------------
                H = build_H_fim_from_paths(
                    params,
                    P_B_ref, Zeta_B_ref,
                    P_U_ref, Zeta_U_ref,
                    wavelength, fs, K
                )

                # --------------------------------------------
                # Pilot transmission
                # --------------------------------------------
                Tpilots = 1

                S_all = generate_multi_pilots(
                    N_U,
                    Tpilots
                )

                Ym, sigma2, _ = pilot_transmission_multi(
                    H,
                    N_B,
                    N_U,
                    S_all,
                    SNR_dB,
                    K
                )

                # --------------------------------------------
                # TE / LS estimation
                # --------------------------------------------
                Hhat = np.zeros_like(H, dtype=complex)

                for k in range(K):

                    # Shape:
                    # (N_B, N_U, Tpilots)
                    Yk_all = Ym[:, :, k, :]

                    # Classical LS:
                    # lambda_r = 0
                    Hhat[:, :, k] = ridge_TE_multipilot(
                        Yk_all,
                        S_all,
                        lambda_r=sigma2
                    )

                # --------------------------------------------
                # NMSE accumulation
                # --------------------------------------------
                err_sum += np.linalg.norm(
                    Hhat - H
                ) ** 2

                pow_sum += np.linalg.norm(H) ** 2

            NMSE_LS[iL, iS] = err_sum / pow_sum

            print(
                f"L={L}, "
                f"SNR={SNR_dB} dB, "
                f"NMSE={NMSE_LS[iL, iS]:.4e}"
            )

    return NMSE_LS
