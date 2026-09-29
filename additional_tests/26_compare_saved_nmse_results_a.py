"""Exported verbatim from notebook code cell 26."""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SNR AXIS
# ============================================================
SNR_dB = np.array([-10, -5, 0, 5, 10, 15, 20])

# ============================================================
# OLD 200-EPOCH RESULT (L = 3)
# From first figure
# ============================================================
NMSE_old_L3 = np.array([
    0.090613,
    0.0293543,
    0.01048299,
    0.00430095,
    0.00204493,
    0.00099399,
    0.000649931
])

# ============================================================
# NEW RESULT (L = 3)
# From second image
# ============================================================
NMSE_new_L3 = np.array([0.39319082, 
                         0.13425748 ,
                         0.048664,
                         0.01897541,
                         0.00875661,
                         0.00492187,
                         0.00339052])


# ============================================================
# LS CURVE (L = 3)
# ============================================================
NMSE_LS = np.array([
    9.53107856,
    3.11534433,
    0.99645222,
    0.31596920,
    0.09985185,
    0.03161006,
    0.00998320
])

# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(7.2,5.2))

plt.semilogy(
    SNR_dB,
    NMSE_LS,
    '-*',
    linewidth=1.5,
    markersize=7,
    label='LS (L=3)'
)

plt.semilogy(
    SNR_dB,
    NMSE_old_L3,
    '-s',
    linewidth=1.5,
    markersize=6,
    markerfacecolor='white',
    label='SF-CNN, L=3 (Multiple deformations)'
)

plt.semilogy(
    SNR_dB,
    NMSE_new_L3,
    '-o',
    linewidth=1.5,
    markersize=6,
    markerfacecolor='white',
    label='SF-CNN, L=3 (Rigid array)'
)

# ============================================================
# STYLE
# ============================================================
plt.grid(True, which='both', linestyle='-', alpha=0.4)

plt.xlabel('SNR (dB)', fontsize=11)
plt.ylabel('NMSE', fontsize=11)

plt.title(
    'Evaluation of SF-CNN, NMSE vs SNR',
    fontsize=12
)

plt.xlim([-10, 20])
plt.ylim([1e-4, 1e1])

plt.legend(fontsize=9, loc='lower left')

plt.tight_layout()
plt.show()
