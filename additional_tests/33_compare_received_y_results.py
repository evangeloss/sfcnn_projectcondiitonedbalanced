"""Exported verbatim from notebook code cell 33."""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SNR AXIS
# ============================================================
SNR_dB =([-10,
                   -5,
                   0,
                   5,
                   10,
                   15,
                   20
                  ])

# ============================================================
# OLD 200-EPOCH RESULT (L = 3)
# From first figure
# ============================================================
NMSE_old_L3 = ([
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
NMSE_new_L3 = ([0.39319082, 
                         0.13425748 ,
                         0.048664,
                         0.01897541,
                         0.00875661,
                         0.00492187,
                         0.00339052])


# ============================================================
# LS CURVE (L = 3)
# ============================================================
NMSE_LS = ([
    9.53107856,
    3.11534433,
    0.99645222,
    0.31596920,
    0.09985185,
    0.03161006,
    0.00998320
])

NMSE_OMP = ([ 0.2092,
                     0.1701,
                     0.1418,
                     0.1296,
                     0.1447,
                     0.1394,
                     0.1335])

NMSE_LMMSE  = ([
    0.1203,      # -9.1952 dB
    0.0491,      # -13.0944 dB
    0.0225,      # -16.4866 dB
    0.00952,
    0.00473,     # -23.2453 dB
    0.00169,     # -27.7143 dB
    0.000648     # -31.8819 dB
])

NMSE_CNN = ( [
    0.64656693,
    0.34210689,
    0.14880911,
    0.11285083,
    0.09058311,
    0.09331958,
    0.10234478
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
    label='LS'
)

plt.semilogy(
    SNR_dB,
    NMSE_old_L3,
    '-s',
    linewidth=1.5,
    markersize=6,
    markerfacecolor='white',
    label='SF-CNN (Multiple deformations)'
)

plt.semilogy(
    SNR_dB,
    NMSE_new_L3,
    '-o',
    linewidth=1.5,
    markersize=6,
    markerfacecolor='white',
    label='SF-CNN (Rigid array)'
)

plt.semilogy(
    SNR_dB,
    NMSE_OMP,
    '-o',
    linewidth=1.5,
    markersize=6,
    markerfacecolor='white',
    label='OMP'
)

plt.semilogy(
    SNR_dB,
    NMSE_LMMSE,
    '-o',
    linewidth=1.5,
    markersize=6,
    markerfacecolor='white',
    label='LMMSE'
)

plt.semilogy(
    SNR_dB,
    NMSE_CNN,
    '-o',
    linewidth=1.5,
    markersize=6,
    markerfacecolor='white',
    label='Simple CNN'
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
