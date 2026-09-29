"""Exported verbatim from notebook code cell 5."""

def steering_vector_fim(p0, zeta, wavelength, phi, theta):
    """
    Robust steering vector for FIM.
    Accepts p0 and zeta as either:
        (3, N)
        (N, 3)
    Returns:
        a : (N,)
    """

    p0 = np.asarray(p0, dtype=float)
    zeta = np.asarray(zeta, dtype=float)

    # --------------------------------------------------------
    # Force p0 to shape (3, N)
    # --------------------------------------------------------
    if p0.shape[0] == 3:
        pass
    elif p0.shape[1] == 3:
        p0 = p0.T
    else:
        raise ValueError(f"p0 must be shape (3,N) or (N,3), got {p0.shape}")

    # --------------------------------------------------------
    # Force zeta to shape (3, N)
    # --------------------------------------------------------
    if zeta.shape[0] == 3:
        pass
    elif zeta.shape[1] == 3:
        zeta = zeta.T
    else:
        raise ValueError(f"zeta must be shape (3,N) or (N,3), got {zeta.shape}")

    assert p0.shape == zeta.shape, f"p0 and zeta shape mismatch: {p0.shape} vs {zeta.shape}"

    N = p0.shape[1]

    # Total deformed coordinates
    p = p0 + zeta  # shape: (3, N)

    # Direction vector
    u = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ]).reshape(3)

    # Phase
    phase = (2 * np.pi / wavelength) * (u @ p)  # shape: (N,)

    # Steering vector
    a = np.exp(1j * phase) / np.sqrt(N)

    return a
