"""Exported verbatim from notebook code cell 46."""

import numpy as np
N_B = 25 
N_U = 25
fc = 28e9 
c = 3e8
wavelength = c/fc

# Deformation normal displacement
b = 1.0 * wavelength   # b/lambda = 1

xi_B = b * (2*np.random.rand(N_B) - 1)
xi_U = b * (2*np.random.rand(N_U) - 1)
print(xi_B)
print(np.max(np.abs(xi_B)) / wavelength)
