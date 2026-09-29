"""Exported verbatim from notebook code cell 44."""

def print_codebook_morphing_range(Zeta_B_all, Zeta_U_all, wavelength):
    print("Original codebook deformation ranges:")

    for m in range(len(Zeta_B_all)):
        max_B = np.max(np.linalg.norm(Zeta_B_all[m], axis=0)) / wavelength
        max_U = np.max(np.linalg.norm(Zeta_U_all[m], axis=0)) / wavelength

        print(
            f"m={m+1}: "
            f"max BS ||zeta||/lambda = {max_B:.4f}, "
            f"max UE ||zeta||/lambda = {max_U:.4f}"
        )

print_codebook_morphing_range(Zeta_B_all, Zeta_U_all, wavelength)
