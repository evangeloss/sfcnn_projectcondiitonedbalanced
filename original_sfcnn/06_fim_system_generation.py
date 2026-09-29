"""Exported verbatim from notebook code cell 6."""

def generateFIMsystem(
    NH_B, NV_B, dxB, dyB,
    NH_U, NV_U, dxU, dyU,
    fc, flag=False
):
    """
    Python conversion of generateFIMsystem.m

    Returns
    -------
    P_B_mat : ndarray, shape (3, N_B)
        BS-side rigid FIM coordinates.
    Zeta_B : ndarray, shape (3, N_B)
        BS-side deformation matrix.
    P_U_mat : ndarray, shape (3, N_U)
        UE-side rigid FIM coordinates.
    Zeta_U : ndarray, shape (3, N_U)
        UE-side deformation matrix.
    """

    wavelength = 3e8 / fc

    N_B = NH_B * NV_B
    N_U = NH_U * NV_U

    # Centers
    pc_B = np.array([0.0, 0.0, 0.0])
    pc_U = np.array([5 * wavelength, 0.0, 0.0])

    # Orientations
    az_B = 0.0
    el_B = np.pi / 2
    roll_B = 0.0

    az_U = 0.0
    el_U = -np.pi / 2
    roll_U = 0.0

    alpha = 0.02
    b = alpha * wavelength
    xi_B = b * (2*np.random.rand(N_B) - 1)
    xi_U = b * (2*np.random.rand(N_U) - 1)
    # Generate BS geometry
    P_B_def, P_B, xi_B_vec, it_B, jt_B, kt_B = generate_FIM_deformed(
        NH_B, NV_B, dxB, dyB,
        pc_B, az_B, el_B, roll_B,
        xi_B
    )

    # Generate UE geometry
    P_U_def, P_U, xi_U_vec, it_U, jt_U, kt_U = generate_FIM_deformed(
        NH_U, NV_U, dxU, dyU,
        pc_U, az_U, el_U, roll_U,
        xi_U
    )

    # Convert to steering-vector format: 3 x N
    P_B_mat = P_B.T
    P_U_mat = P_U.T

    # zeta = xi_n * k_t
    Zeta_B = kt_B[:, None] * xi_B_vec[None, :]
    Zeta_U = kt_U[:, None] * xi_U_vec[None, :]

    if flag:
        try:
            plot_fim_system(
                P_B, P_B_def,
                P_U, P_U_def,
                NH_B, NV_B, NH_U, NV_U
            )
        except NameError:
            print("plot_fim_system is not defined.")

    return P_B_mat, Zeta_B, P_U_mat, Zeta_U
