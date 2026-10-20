"""Checks of existing numerical and analytic claims in chapter rc."""
import mpmath as mp
import numpy as np
from math import comb, factorial, log10, log
mp.mp.dps = 40

# 1. E log alpha sign in the Mellin theorem: R = 1/((a+1)(a+2)(a+3)), c = 1
r = [mp.mpf(1), mp.mpf(2), mp.mpf(3)]
gam = [mp.mpf(1) / 2, mp.mpf(-1), mp.mpf(1) / 2]  # residues of 1/((x+1)(x+2)(x+3))
R = lambda x: 1 / ((x + 1) * (x + 2) * (x + 3))
C = mp.quad(R, [0, mp.inf])
El = mp.quad(lambda x: mp.log(x) * R(x), [0, mp.inf]) / C
S1 = sum(g * mp.log(q) for g, q in zip(gam, r))
S2 = sum(g * mp.log(q) ** 2 for g, q in zip(gam, r))
print("C quad", C, "-sum g log r", -S1)
print("E log quad", El, " S2/(2 S1)", S2 / (2 * S1), " S2/(-2 S1)", S2 / (-2 * S1))

# 2. Example double: Ewens n=20,k=5 with PH(0,1,2,1)
n, k = 20, 5
f = lambda x: x ** (k - 1) / ((x + 1) ** 2 * (x + 2) ** 2 * mp.fprod([x + i for i in range(3, n)]))
print("double example quad", mp.quad(f, [0, 1, 10, 100, mp.inf]))

# 3. Prop cond (i): log10 kappa_n for k=1
for n in [10, 20, 40, 80, 160]:
    mp.mp.dps = 120
    Rn = lambda x: mp.gamma(x + 1) / mp.gamma(x + n)
    I = mp.quad(Rn, [0, 1, 10, mp.inf])
    gmax = mp.mpf(comb(n - 2, (n - 2) // 2)) / mp.factorial(n - 2)
    kap = gmax / I
    asym_nologlog = n * log10(2) + 0.5 * log10(n)
    print(n, "log10 kappa", float(mp.log10(kap)), " n log2 + .5 log n:", round(asym_nologlog, 3),
          " +log10 log n:", round(asym_nologlog + log10(log(n)), 3))
mp.mp.dps = 40

# 4. Prop cond (ii): x*, Phi(x*) for rho = 1/8
rho = mp.mpf(1) / 8
xs = mp.findroot(lambda x: rho / x - mp.log(x / (1 - x)), 0.5)
Phi = lambda x: rho * mp.log(x) - x * mp.log(x) - (1 - x) * mp.log(1 - x) + 1
print("x*", xs, "Phi", Phi(xs))
n = 1600; k = n // 8
lg = [(k - 1) * mp.log(j) - mp.loggamma(j) - mp.loggamma(n - j) for j in range(1, n)]
ex = max(lg)
print("exact max log gamma", ex, "formula", (rho - 1) * n * mp.log(n) + Phi(xs) * n,
      "rel", (ex - ((rho - 1) * n * mp.log(n) + Phi(xs) * n)) / ex)

# 5. Erlang c=4 zeros; and when do zeros enter the right half plane
for c in range(2, 11):
    coeffs = [1 / factorial(j) for j in range(c, -1, -1)]
    z = np.roots(coeffs)
    print("c", c, "max real part", max(z.real).round(4), (np.sort_complex(z)).round(4) if c == 4 else "")

# 6. Corollary rate: int_0^delta p for PH(1,1,3,1) scaling
for m in [0, 1, 2, 3]:
    b = m + 2
    p = lambda x: mp.rf(x, m) / mp.rf(x + 1, b)
    d1, d2 = mp.mpf("1e-4"), mp.mpf("1e-5")
    I1 = mp.quad(p, [0, d1]); I2 = mp.quad(p, [0, d2])
    print("m", m, "local exponent", mp.log(I1 / I2) / mp.log(10))

# 7. Verification table rows
f1 = lambda x: (x + mp.mpf(1) / 2) / ((x + 1) ** 2 * (x + 2) ** 2 * (x + 3))
print("row1", mp.quad(f1, [0, mp.inf]))
f2 = lambda x: 1 / ((x + 1) * ((x + 2) ** 2 + 9))
print("row2", mp.quad(f2, [0, mp.inf]))
