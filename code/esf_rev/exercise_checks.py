"""Numerical answers quoted in the exercises."""
import numpy as np
from scipy.special import gammaln
n = 3306
# finite pool expected number of species
m, a = 735, 0.4936
EK = m * (1 - np.exp(gammaln((m - 1) * a + n) - gammaln((m - 1) * a) - gammaln(m * a + n) + gammaln(m * a)))
print("finite pool E K_n", EK)
# Ewens new species in a second expedition
th = 163.994
i = np.arange(n, 2 * n)
print("Ewens E new", (th / (th + i)).sum(), "theta log 2", th * np.log(2))
