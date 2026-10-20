"""Exact law of S for n = 5 as a sum of four independent geometrics; check by simulation."""
import numpy as np
from fractions import Fraction as Fr
n = 5
def pmf(theta, smax):
    p = np.zeros(smax + 1); p[0] = 1
    for j in range(1, n):
        q = theta / (theta + j)
        p = np.convolve(p, (1 - q) * q ** np.arange(smax + 1))[: smax + 1]
    return p
an = sum(1 / j for j in range(1, n)); bn = sum(1 / j**2 for j in range(1, n))
print("a5", an, Fr(25, 12), "b5", bn, Fr(205, 144))
for th in (1.0, 3.0):
    p = pmf(th, 400)
    s = np.arange(401)
    print("theta", th, "P(S=s) s=0..8", np.round(p[:9], 4), "P(S>=9)", round(1 - p[:9].sum(), 4))
    print("  mean", (s * p).sum(), th * an, "var", (s * s * p).sum() - (s * p).sum() ** 2, th * an + th**2 * bn)
# exact P(S=0) = prod j/(theta+j), at theta=1: 1/5
print("P(S=0|1)", Fr(1*2*3*4, 2*3*4*5))
# P(S=1 | theta) closed form: P(S=0) * sum_j theta/(theta+j)
for th in (1.0, 3.0):
    p0 = np.prod([j / (th + j) for j in range(1, n)])
    print("P(S=1) closed", p0 * sum(th / (th + j) for j in range(1, n)))
# simulation by coalescent tree with Poisson mutations
rng = np.random.default_rng(1)
R = 400000
for th in (1.0, 3.0):
    S = np.zeros(R, int)
    for k in range(n, 1, -1):
        T = rng.exponential(1 / (k * (k - 1) / 2), R)
        S += rng.poisson(th * k * T / 2)
    print("sim theta", th, np.round(np.bincount(S, minlength=9)[:9] / R, 4))
# posterior given S under the log uniform prior on [0.5,30], for S = 0..6
from scipy.integrate import quad
for s in range(0, 7):
    f = lambda t: pmf(t, s)[s] / t
    Z = quad(f, 0.5, 30, limit=200)[0]
    from scipy.optimize import brentq
    q = [brentq(lambda x: quad(f, 0.5, x, limit=200)[0] / Z - a, 0.5, 30) for a in (0.05, .5, .95)]
    print("S", s, "post q05,med,q95", np.round(q, 2), "Watterson", round(s / an, 2))
