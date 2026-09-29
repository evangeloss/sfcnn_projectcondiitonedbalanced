"""Exported verbatim from notebook code cell 34."""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ACHIEVABLE RATE FROM TRUE AND ESTIMATED CHANNEL
# ============================================================

def achievable_rate_svd(H_true, H_beam, SNR_dB, Ns=1):
    """
    H_true : [N_B, N_U, K]
        True physical channel used for rate evaluation.

    H_beam : [N_B, N_U, K]
        Channel used to design the precoder.
        For perfect CSI: H_beam = H_true
        For estimated CSI: H_beam = H_est

    SNR_dB : scalar
    Ns     : number of streams

    Returns:
        Average achievable rate over K subcarriers.
    """

    N_B, N_U, K = H_true.shape
    rho = 10 ** (SNR_dB / 10)

    rate_sum = 0.0

    for k in range(K):

        Hk = H_true[:, :, k]
        Hhat_k = H_beam[:, :, k]

        # SVD of estimated/beamforming channel
        _, _, Vh = np.linalg.svd(Hhat_k, full_matrices=False)

        # Precoder from right singular vectors
        F = Vh.conj().T[:, :Ns]

        # Normalize total transmit power
        F = F / np.linalg.norm(F, "fro") * np.sqrt(Ns)

        # Rate evaluated on true channel
        A = np.eye(N_B, dtype=np.complex128) + \
            (rho / Ns) * Hk @ F @ F.conj().T @ Hk.conj().T

        sign, logdet = np.linalg.slogdet(A)

        rate_k = np.real(logdet / np.log(2))

        rate_sum += rate_k

    return rate_sum / K


# ============================================================
# RATE VS SNR
# ============================================================

def rate_vs_snr(H_true_all, H_est_all, SNR_vec_dB, Ns=1):
    """
    H_true_all : [Ntest, K, N_B, N_U]
    H_est_all  : [Ntest, K, N_B, N_U]
    """

    Ntest = H_true_all.shape[0]

    Rate_perfect = np.zeros(len(SNR_vec_dB))
    Rate_est = np.zeros(len(SNR_vec_dB))

    for iSNR, SNR_dB in enumerate(SNR_vec_dB):

        print(f"Evaluating rate at SNR = {SNR_dB} dB")

        rate_perf_sum = 0.0
        rate_est_sum = 0.0

        for n in range(Ntest):

            # Convert from [K, N_B, N_U] to [N_B, N_U, K]
            H_true = np.transpose(H_true_all[n], (1, 2, 0))
            H_est = np.transpose(H_est_all[n], (1, 2, 0))

            # Perfect CSI beamforming
            rate_perf_sum += achievable_rate_svd(
                H_true,
                H_true,
                SNR_dB,
                Ns=Ns
            )

            # Estimated CSI beamforming
            rate_est_sum += achievable_rate_svd(
                H_true,
                H_est,
                SNR_dB,
                Ns=Ns
            )

        Rate_perfect[iSNR] = rate_perf_sum / Ntest
        Rate_est[iSNR] = rate_est_sum / Ntest

    return Rate_perfect, Rate_est


# ============================================================
# EXAMPLE USAGE
# ============================================================

SNR_vec_dB = np.arange(-10, 25, 5)
Ns = 1

Rate_perfect, Rate_est = rate_vs_snr(
    H_true_all,
    H_est_all,
    SNR_vec_dB,
    Ns=Ns
)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(7.2, 5.2))

plt.plot(
    SNR_vec_dB,
    Rate_perfect,
    "-o",
    linewidth=1.7,
    markersize=6,
    markerfacecolor="white",
    label="Perfect CSI"
)

plt.plot(
    SNR_vec_dB,
    Rate_est,
    "-s",
    linewidth=1.7,
    markersize=6,
    markerfacecolor="white",
    label="Estimated CSI"
)

plt.grid(True, linestyle="-", alpha=0.4)
plt.xlabel("SNR (dB)")
plt.ylabel("Achievable Rate (bits/s/Hz)")
plt.title("Achievable Rate vs SNR")
plt.legend()
plt.tight_layout()
plt.show()

print("Rate_perfect =")
print(Rate_perfect)

print("Rate_est =")
print(Rate_est)
