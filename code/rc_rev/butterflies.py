# NOTE (round 3): this script reproduces the UNCENSORED analysis (the 24 tabulated classes taken
# as the whole sample, n = 3306, K = 501, theta hat 163.99), which chapter rc now quotes only as the
# "ignoring the censoring" comparison (Section on Corbet's butterflies, Example rc:ex:butterflies).
# The censored analysis the chapter uses (n hat = 11068, K = 620) is in butterflies_censored.py
# and butterflies_censored_residue.py.
"""Corbet's Malayan butterflies (Fisher, Corbet and Williams 1943), abundance classes j=1..24,
as in Chapter esf: n = 3306 individuals, K = 501 species. Posterior of the Ewens theta
under PH(0,1,2,1), Ga(2,1), Ga(0.1,0.1): stable log domain quadrature, and for the half
horseshoe also the exact residue formula in high precision arithmetic."""
import numpy as np
import mpmath as mp
from scipy.special import gammaln
from scipy.integrate import quad
import time

a = np.array([118, 74, 44, 24, 29, 22, 20, 19, 20, 15, 12, 14,
              6, 12, 6, 9, 9, 6, 10, 10, 11, 5, 3, 3])
j = np.arange(1, 25)
n = int((a * j).sum()); K = int(a.sum())
print("n", n, "K", K)

def loglik(t):
    return K * np.log(t) + gammaln(t) - gammaln(t + n)

priors = {
    "PH(0,1,2,1)": lambda t: -np.log(t + 1) - np.log(t + 2),
    "Ga(2,1)": lambda t: np.log(t) - t,
    "Ga(0.1,0.1)": lambda t: -0.9 * np.log(t) - 0.1 * t,
}
grid = np.linspace(1e-6, 600, 600001)
res = {}
for name, lp in priors.items():
    lg = loglik(grid) + lp(grid)
    mx = lg.max()
    w = np.exp(lg - mx)
    cdf = np.cumsum(w); cdf /= cdf[-1]
    mean = np.sum(grid * w) / np.sum(w)
    q = [grid[np.searchsorted(cdf, p)] for p in (0.025, 0.5, 0.975)]
    pred = np.sum(grid / (grid + n) * w) / np.sum(w)
    res[name] = (mean, q, pred)
    print("%-12s mean %.2f median %.2f  95%% (%.2f, %.2f)  P(new) %.5f  mode %.2f" %
          (name, mean, q[1], q[0], q[2], pred, grid[np.argmax(lg)]))

# MLE
from scipy.optimize import brentq
from scipy.special import digamma
mle = brentq(lambda t: t * (digamma(t + n) - digamma(t)) - K, 1, 2000)
print("MLE", mle, "plug-in P(new)", mle / (mle + n))
I = sum((i - 1) / (mle * (mle + i - 1) ** 2) for i in range(1, n + 1))
print("Fisher info at MLE", I, "se", I ** -0.5, "log n/theta", np.log(n) / mle)

# exact residues for the half horseshoe, kernel theta^p / [(t+1)^2 (t+2)^2 prod_{i=3}^{n-1}(t+i) (t+n)^e]
def residue_int(n, p, e):
    lf = mp.log  # noqa
    tot = mp.mpf(0)
    gsum = mp.mpf(0)
    gmax = mp.mpf(0)
    # double pole at -1
    h1 = (-1) ** p / (mp.factorial(n - 2) * (mp.mpf(n - 1) ** e))
    d1 = h1 * (mp.mpf(p) / (-1) - 2 - mp.fsum(mp.mpf(1) / (i - 1) for i in range(3, n)) - e * mp.mpf(1) / (n - 1))
    # double pole at -2
    h2 = mp.mpf(-2) ** p / (mp.factorial(n - 3) * (mp.mpf(n - 2) ** e))
    d2 = h2 * (mp.mpf(p) / (-2) + 2 - mp.fsum(mp.mpf(1) / (i - 2) for i in range(3, n)) - e * mp.mpf(1) / (n - 2))
    tot += -d1 * mp.log(1) + h1 / 1 - d2 * mp.log(2) + h2 / 2
    gsum += d1 + d2
    gmax = max(abs(d1), abs(d2), abs(h1), abs(h2))
    for jj in range(3, n):
        den = mp.mpf(1 - jj) ** 2 * mp.mpf(2 - jj) ** 2 * (-1) ** (jj - 3) * mp.factorial(jj - 3) * mp.factorial(n - 1 - jj)
        if e:
            den *= (n - jj)
        g = mp.mpf(-jj) ** p / den
        tot += -g * mp.log(jj)
        gsum += g
        gmax = max(gmax, abs(g))
    if e:
        den = mp.mpf(n - 1) ** 2 * mp.mpf(n - 2) ** 2 * (-1) ** (n - 3) * mp.factorial(n - 3)
        g = mp.mpf(-n) ** p / den
        tot += -g * mp.log(n)
        gsum += g
        gmax = max(gmax, abs(g))
    return tot, gsum, gmax

# test on Example double: n=20, k=5
mp.mp.dps = 50
print("test n=20,k=5:", residue_int(20, 4, 0)[0])

for dps in [1500, 2500]:
    mp.mp.dps = dps
    t0 = time.time()
    C, gs, gm = residue_int(n, K - 1, 0)
    M1, _, _ = residue_int(n, K, 0)
    P, _, _ = residue_int(n, K, 1)
    print("dps", dps, "time %.1f s" % (time.time() - t0))
    print("  C", mp.nstr(C, 15), " log10 kappa", mp.nstr(mp.log10(gm / C), 6), " sum gamma_i1", mp.nstr(gs, 3))
    print("  mean", mp.nstr(M1 / C, 15), " P(new)", mp.nstr(P / C, 15))
# compare C with quadrature in log domain
lC_quad = None
f = lambda t: np.exp(loglik(t) + priors["PH(0,1,2,1)"](t) - (K - 1) * 0)  # not used
mp.mp.dps = 60
g = lambda t: mp.exp(K * mp.log(t) + mp.loggamma(t) - mp.loggamma(t + n)) / ((t + 1) * (t + 2))
q = mp.quad(g, [0, 50, 100, 130, 150, 160, 170, 180, 200, 250, 400, mp.inf])
print("quad C (mp)", mp.nstr(q, 15))
q1 = mp.quad(lambda t: t * g(t), [0, 50, 100, 130, 150, 160, 170, 180, 200, 250, 400, mp.inf])
print("quad mean", mp.nstr(q1 / q, 15))
