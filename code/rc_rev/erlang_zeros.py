"""Zeros of the exponential sections sum_{j<=c} A^j/j!: number of real zeros, largest real part."""
import numpy as np
from math import factorial
for c in range(1, 13):
    z = np.roots([1 / factorial(j) for j in range(c, -1, -1)])
    nreal = int(np.sum(np.abs(z.imag) < 1e-9))
    print(c, "real zeros", nreal, "max Re %.4f" % z.real.max(), np.sort_complex(z).round(4) if c == 4 else "")
