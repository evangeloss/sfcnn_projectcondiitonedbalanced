"""Exported verbatim from notebook code cell 25."""

plt.figure(figsize=(6, 4))

plt.plot(
    L_list,
    10*np.log10(NMSE_TE_avgSNR),
    'o-',
    linewidth=2,
    label='TE / LS'
)

plt.plot(
    L_list,
    10*np.log10(NMSE_CNN_avgSNR),
    's-',
    linewidth=2,
    label='SF-CNN'
)

plt.xlabel('Number of propagation paths, L')
plt.ylabel('Average NMSE over SNRs (dB)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
