"""Exported verbatim from notebook code cell 36."""

import numpy as np
import pickle
import os

# ============================================================
# 1) Observation model for one deformation codebook
# ============================================================

def fim_observation_vector(
    params,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    S,
    fc, fs, K
):
    """
    Stacks all received noiseless pilot observations:

        y = [vec(H_1[k] S), ..., vec(H_M[k] S)]

    This is the mean vector μ(θ) used for the FIM.
    """

    wavelength = 3e8 / fc
    M = len(P_B_all)

    y_list = []

    for m in range(M):

        Hm = build_H_fim_from_paths(
            params,
            P_B_all[m], Zeta_B_all[m],
            P_U_all[m], Zeta_U_all[m],
            wavelength, fs, K
        )

        for k in range(K):
            Ymk = Hm[:, :, k] @ S
            y_list.append(Ymk.reshape(-1, order="F"))

    return np.concatenate(y_list)


# ============================================================
# 2) Pack / unpack path parameters
# ============================================================

def pack_params(params):
    """
    theta per path:
        [Re(beta), Im(beta), delay, AOA_az, AOA_el, DOA_az, DOA_el]
    """

    L = len(params["BETA"])
    theta = []

    for l in range(L):
        theta.extend([
            np.real(params["BETA"][l]),
            np.imag(params["BETA"][l]),
            params["delay"][l],
            params["AOA_az"][l],
            params["AOA_el"][l],
            params["DOA_az"][l],
            params["DOA_el"][l]
        ])

    return np.array(theta, dtype=float)


def unpack_params(theta, template_params):
    """
    Converts theta vector back to your notebook params dictionary.
    """

    L = len(theta) // 7

    params = {}
    params["BETA"] = np.zeros(L, dtype=complex)
    params["delay"] = np.zeros(L)
    params["AOA_az"] = np.zeros(L)
    params["AOA_el"] = np.zeros(L)
    params["DOA_az"] = np.zeros(L)
    params["DOA_el"] = np.zeros(L)

    for l in range(L):
        ar, ai, tau, aoa_az, aoa_el, doa_az, doa_el = theta[7*l:7*(l+1)]

        params["BETA"][l] = ar + 1j * ai
        params["delay"][l] = tau
        params["AOA_az"][l] = aoa_az
        params["AOA_el"][l] = aoa_el
        params["DOA_az"][l] = doa_az
        params["DOA_el"][l] = doa_el

    return params


# ============================================================
# 3) Fisher Information Matrix
# ============================================================

def compute_fim_for_codebook(
    params,
    P_B_all, Zeta_B_all,
    P_U_all, Zeta_U_all,
    S,
    fc, fs, K,
    sigma2,
    eps=1e-5
):
    """
    Complex Gaussian model:

        y = μ(θ) + n
        n ~ CN(0, sigma2 I)

    FIM:

        J = 2/sigma2 Re{D^H D}
    """

    theta0 = pack_params(params)
    n_params = len(theta0)

    def mu_from_theta(theta):
        p = unpack_params(theta, params)
        return fim_observation_vector(
            p,
            P_B_all, Zeta_B_all,
            P_U_all, Zeta_U_all,
            S,
            fc, fs, K
        )

    mu0 = mu_from_theta(theta0)
    D = np.zeros((len(mu0), n_params), dtype=complex)

    for i in range(n_params):

        step = eps * max(1.0, abs(theta0[i]))

        theta_plus = theta0.copy()
        theta_minus = theta0.copy()

        theta_plus[i] += step
        theta_minus[i] -= step

        mu_plus = mu_from_theta(theta_plus)
        mu_minus = mu_from_theta(theta_minus)

        D[:, i] = (mu_plus - mu_minus) / (2 * step)

    J = (2 / sigma2) * np.real(D.conj().T @ D)

    return J, D



def score_fim_codebook(J, eps_reg=1e-9):
    """
    Main objective:

        maximize log det(J)

    Also returns CRB trace.
    """

    Jreg = J + eps_reg * np.eye(J.shape[0])

    sign, logdetJ = np.linalg.slogdet(Jreg)

    if sign <= 0:
        logdetJ = -np.inf

    CRB = np.linalg.pinv(Jreg)
    trCRB = np.real(np.trace(CRB))

    condJ = np.linalg.cond(Jreg)

    return logdetJ, trCRB, condJ
