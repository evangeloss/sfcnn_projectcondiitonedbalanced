"""Exported verbatim from notebook code cell 24."""

L_list, NMSE_TE_avgSNR, NMSE_CNN_avgSNR = NMSE_CNN_vs_paths_avgSNR(
    N_B, N_U, fc, fs, K,
    NchCNN=200,
    P_B_all=P_B_all,
    Zeta_B_all=Zeta_B_all,
    P_U_all=P_U_all,
    Zeta_U_all=Zeta_U_all,
    M=M,
    netSF=netSF,
    SNR_vec_dB=np.array([-10, -5, 0, 5, 10, 15, 20]),
    L_list=np.arange(1, 11),
    Tpilots=1,
    device=device
)
