"""Exported verbatim from notebook code cell 3."""

def generate_FIM_deformed(NH, NV, dx, dy, pc, az, el, roll, xi):
    """
    Python conversion of generate_FIM_deformed.m

    Parameters
    ----------
    NH, NV : int
        Number of horizontal and vertical FIM elements.
    dx, dy : float
        Element spacing in x and y directions.
    pc : array_like, shape (3,)
        Center position of the FIM.
    az, el, roll : float
        Rotation angles in radians.
    xi : array_like
        Deformation displacement.
        Can be:
        - shape (N,), where N = NH * NV
        - shape (NV, NH)

    Returns
    -------
    P_def : ndarray, shape (N, 3)
        Deformed element positions.
    P : ndarray, shape (N, 3)
        Undeformed element positions.
    xi_vec : ndarray, shape (N,)
        Deformation vector.
    it, jt, kt : ndarray, shape (3,)
        Local rotated basis vectors.
    """

    N = NH * NV
    pc = np.asarray(pc, dtype=float).reshape(3)
    xi = np.asarray(xi, dtype=float)

    # Match MATLAB reshape(xi.', [], 1)
    if xi.ndim == 2 and xi.shape == (NV, NH):
        xi_vec = xi.T.reshape(-1)
    elif xi.size == N:
        xi_vec = xi.reshape(-1)
    else:
        raise ValueError("xi must be either NH*NV vector or NV-by-NH matrix.")

    Rz = np.array([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az),  np.cos(az), 0],
        [0,           0,          1]
    ])

    Ry = np.array([
        [ np.cos(el), 0, np.sin(el)],
        [0,           1, 0],
        [-np.sin(el), 0, np.cos(el)]
    ])

    Rx = np.array([
        [1, 0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    R = Rz @ Ry @ Rx

    it = R[:, 0]
    jt = R[:, 1]
    kt = R[:, 2]

    x_local = (np.arange(NH) - (NH - 1) / 2) * dx
    y_local = (np.arange(NV) - (NV - 1) / 2) * dy

    P = np.zeros((N, 3), dtype=float)

    idx = 0
    for nv in range(NV):
        for nh in range(NH):
            pn = pc + x_local[nh] * it + y_local[nv] * jt
            P[idx, :] = pn
            idx += 1

    P_def = P + xi_vec[:, None] * kt[None, :]

    return P_def, P, xi_vec, it, jt, kt
