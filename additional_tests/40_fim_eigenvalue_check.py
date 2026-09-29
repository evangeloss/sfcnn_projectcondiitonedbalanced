"""Exported verbatim from notebook code cell 40."""

eigvals = np.linalg.eigvalsh(J_rand)
print(np.min(eigvals))
print(np.max(eigvals))
