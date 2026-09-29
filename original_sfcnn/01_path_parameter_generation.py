"""Exported verbatim from notebook code cell 1."""

def generate_path_parameters(L, fc, fs):
    """
    Generate random multipath channel parameters.

    Parameters
    ----------
    L : int
        Number of propagation paths.
    fc : float
        Carrier frequency (currently unused, kept for compatibility).
    fs : float
        Sampling frequency.

    Returns
    -------
    params : dict
        Dictionary containing:
        - AOA_az : Azimuth angles of arrival
        - AOA_el : Elevation angles of arrival
        - DOA_az : Azimuth angles of departure
        - DOA_el : Elevation angles of departure
        - BETA   : Complex path gains
        - delay  : Path delays
    """

    params = {}

    # Angles
    params["AOA_az"] = (np.random.rand(L, 1) - 0.5) * 2 * np.pi
    params["AOA_el"] = (np.random.rand(L, 1) - 0.5) * np.pi

    params["DOA_az"] = (np.random.rand(L, 1) - 0.5) * 2 * np.pi
    params["DOA_el"] = (np.random.rand(L, 1) - 0.5) * np.pi

    # Complex path gains
    params["BETA"] = (
        np.random.randn(L, 1) + 1j * np.random.randn(L, 1)
    ) / np.sqrt(2 * L)

    # Delays
    max_delay = 1 / fs
    params["delay"] = np.random.rand(L, 1) * max_delay

    return params
