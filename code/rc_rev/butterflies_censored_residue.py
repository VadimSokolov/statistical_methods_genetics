"""Exact residue formula for the half horseshoe posterior of the Ewens theta on Corbet's
butterflies with censoring (conditional on n = 11068, K = 620; see butterflies_censored.py).
Kernel theta^p / [(t+1)^2 (t+2)^2 prod_{i=3}^{n-1} (t+i) (t+n)^e]; C uses p = K-1, e = 0, the
mean p = K, e = 0, the predictive P(new) p = K, e = 1. Same algebra as residue_int in
butterflies.py, with factorials by recurrence and logs cached across the three integrals.
Usage: python3 -I butterflies_censored_residue.py DPS [n K]"""
import sys
import time
import mpmath as mp

dps = int(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 11068
K = int(sys.argv[3]) if len(sys.argv) > 3 else 620
mp.mp.dps = dps
t0 = time.time()
logs = [mp.mpf(0)] * (n + 1)
for j in range(2, n + 1):
    logs[j] = mp.log(j)
fact = [mp.mpf(1)] * (n + 1)
for j in range(1, n + 1):
    fact[j] = fact[j - 1] * j
H1 = mp.fsum(mp.mpf(1) / (i - 1) for i in range(3, n))
H2 = mp.fsum(mp.mpf(1) / (i - 2) for i in range(3, n))


def residue_int(p, e):
    tot = mp.mpf(0); gsum = mp.mpf(0)
    h1 = (-1) ** p / (fact[n - 2] * (mp.mpf(n - 1) ** e))
    d1 = h1 * (mp.mpf(p) / (-1) - 2 - H1 - e * mp.mpf(1) / (n - 1))
    h2 = mp.mpf(-2) ** p / (fact[n - 3] * (mp.mpf(n - 2) ** e))
    d2 = h2 * (mp.mpf(p) / (-2) + 2 - H2 - e * mp.mpf(1) / (n - 2))
    tot += h1 - d2 * logs[2] + h2 / 2
    gsum += d1 + d2
    gmax = max(abs(d1), abs(d2), abs(h1), abs(h2))
    for jj in range(3, n):
        den = mp.mpf((1 - jj) ** 2 * (2 - jj) ** 2) * fact[jj - 3] * fact[n - 1 - jj]
        if (jj - 3) % 2:
            den = -den
        if e:
            den *= (n - jj)
        g = mp.mpf(-jj) ** p / den
        tot -= g * logs[jj]
        gsum += g
        gmax = max(gmax, abs(g))
    if e:
        den = mp.mpf(n - 1) ** 2 * mp.mpf(n - 2) ** 2 * fact[n - 3]
        if (n - 3) % 2:
            den = -den
        g = mp.mpf(-n) ** p / den
        tot -= g * logs[n]
        gsum += g
        gmax = max(gmax, abs(g))
    return tot, gsum, gmax


C, gs, gm = residue_int(K - 1, 0)
M1, _, _ = residue_int(K, 0)
P, _, _ = residue_int(K, 1)
print("n", n, "K", K, "dps", dps, "time %.1f s" % (time.time() - t0))
print("  C", mp.nstr(C, 15), " log10 kappa", mp.nstr(mp.log10(gm / abs(C)), 6), " sum gamma_i1", mp.nstr(gs, 3))
print("  mean", mp.nstr(M1 / C, 15), " P(new)", mp.nstr(P / C, 15))
