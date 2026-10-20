"""Proposition cond (i): exact log10 kappa_n (k=1) against 2^(n-2) sqrt(2/(pi n)) (n-1) log n."""
import mpmath as mp
from math import comb, log10, log, pi, sqrt
mp.mp.dps = 80
for n in [10, 20, 40, 80, 160]:
    J = mp.quad(lambda x: mp.gamma(x + 1) * mp.gamma(n) / mp.gamma(x + n), [0, 0.1, 1, 10, mp.inf])
    kap = comb(n - 2, (n - 2) // 2) * (n - 1) / J
    asym = (n - 2) * log10(2) + 0.5 * log10(2 / (pi * n)) + log10(n - 1) + log10(log(n))
    print(n, "exact %.3f" % float(mp.log10(kap)), "asym %.3f" % asym, "J*log n %.4f" % float(J * log(n)))
