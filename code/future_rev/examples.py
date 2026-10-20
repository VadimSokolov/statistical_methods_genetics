"""Numerical checks for the illustrative calculations in chapters/future.tex."""
import math
import numpy as np
from scipy import stats

print("=== HW disequilibrium detection sample size")
lam = stats.ncx2.ppf(0.2, 1, 7.849)  # check: power at lambda
crit = stats.chi2.ppf(0.95, 1)
from scipy.optimize import brentq
lam80 = brentq(lambda l: stats.ncx2.sf(crit, 1, l) - 0.8, 0.1, 50)
print("crit", crit, "lambda for 80% power", lam80)
for F in [0.1, 0.03, 0.01]:
    print("F", F, "n", lam80 / F ** 2)

print("=== quadratic functional in normal means")
p, s, b = 10000, 100, 3.0
Q = s * b * b
print("Q", Q, "E sum Y^2", Q + p, "ratio", (Q + p) / Q, "sd unbiased", math.sqrt(2 * p + 4 * Q))

print("=== singleton fraction under PD(sigma, theta), exact expectations")
def moments(sig, th, nmax, checkpoints):
    EK, EK1 = 1.0, 1.0  # n = 1
    out = {}
    for n in range(1, nmax):
        new = (th + sig * EK) / (th + n)
        EK1 = EK1 * (1 - (1 - sig) / (th + n)) + new
        EK = EK + new
        if n + 1 in checkpoints:
            out[n + 1] = (EK, EK1, EK1 / EK)
    return out
cps = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
for sig, th in [(0.5, 0.0), (0.5, 1.0), (0.5, 10.0), (0.25, 1.0)]:
    res = moments(sig, th, 10 ** 6, set(cps))
    print("sigma", sig, "theta", th)
    for n in cps:
        EK, EK1, r = res[n]
        print(f"   n={n:>8d} EK={EK:12.2f} EK1={EK1:12.2f} ratio={r:.4f} diff={r-sig:.4f} diff*n^sig={(r-sig)*n**sig:.3f}")
# Monte Carlo check of recursion at small n
rng = np.random.default_rng(2)
def crp(sig, th, n):
    sizes = [1]
    for m in range(1, n):
        K = len(sizes)
        probs = np.array([c - sig for c in sizes] + [th + sig * K]) / (th + m)
        j = rng.choice(K + 1, p=probs)
        if j == K:
            sizes.append(1)
        else:
            sizes[j] += 1
    return len(sizes), sum(1 for c in sizes if c == 1)
sims = np.array([crp(0.5, 1.0, 100) for _ in range(4000)])
print("MC n=100 sigma .5 theta 1:", sims.mean(0), "exact", moments(0.5, 1.0, 100, {100})[100][:2])
# limiting block size frequencies
sig = 0.5
pj = [sig * math.prod(j - sig for j in range(1, k)) / math.factorial(k) for k in range(1, 6)]
print("limiting fractions sigma .5", pj)

print("=== marginal vs conditional calibration")
# theta ~ N(0,1), X|theta ~ N(theta,1). report prior as posterior.
z = stats.norm.ppf(0.95)
for x in [0, 1, 2, 3]:
    m, sd = x / 2, math.sqrt(0.5)
    cov = stats.norm.cdf((z - m) / sd) - stats.norm.cdf((-z - m) / sd)
    print("x", x, "conditional coverage of prior 90% interval", cov)
# marginal coverage
print("marginal", stats.norm.cdf(z) - stats.norm.cdf(-z))
# Monte Carlo: SBC rank of theta under 'posterior' = prior
th = rng.standard_normal(200000); X = th + rng.standard_normal(200000)
u = stats.norm.cdf(th)  # PIT under prior
print("PIT mean/var", u.mean(), u.var(), "uniform var", 1 / 12)
# conditional on |X|>2
sel = np.abs(X) > 2
print("frac |X|>2", sel.mean(), "coverage there", (np.abs(th[sel]) < z).mean())

print("=== birthday collisions in one WF generation")
for twoN, n in [(20000, 1000), (20000, 100), (20000, 5000), (200000, 1000)]:
    lam = n * (n - 1) / 2 / twoN
    pnone = math.exp(sum(math.log1p(-i / twoN) for i in range(n)))
    edist = twoN * (1 - (1 - 1 / twoN) ** n)
    print(f"2N={twoN} n={n}: expected pair collisions {lam:.3f}, P(no collision) {pnone:.3e}, E distinct parents {edist:.1f}, lost {n-edist:.1f}")
# Kingman: probability at most one merger, expected mergers per generation
