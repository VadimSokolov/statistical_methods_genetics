"""Numbers for the new worked examples in the pig chapter. python -I examples.py"""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np, sys, itertools
from piglib import *
from scipy import special as sp, integrate, stats, optimize
from math import factorial, lgamma
print("=== A. ESF n=4 theta=1 and theta=2")
parts = [(4,0,0,0),(2,1,0,0),(0,2,0,0),(1,0,1,0),(0,0,0,1)]
for th in (1.0, 2.0):
    rf = np.prod([th + i for i in range(4)])
    tot = 0; EKs = 0
    for a in parts:
        p = factorial(4) / rf * np.prod([(th / (j + 1)) ** a[j] / factorial(a[j]) for j in range(4)])
        tot += p; EKs += p * sum(a)
        print(th, a, "K=", sum(a), round(p, 5), p * rf)
    print("sum", tot, "EK", EKs, EK(th, 4), "Var", VarK(th, 4))
# Good check by hand n=4 theta=1
th = 1.0
mu = lambda r, n: th / r * factorial(n) / factorial(n - r) * np.exp(sp.gammaln(th + n - r) - sp.gammaln(th + n))
for r in (1, 2, 3):
    print("mu", r, 4, mu(r, 4), "mu", r + 1, 5, mu(r + 1, 5), "good", (r + 1) / 5 * mu(r + 1, 5) / mu(r, 4), r / (4 + th))
print("=== B. Beta identity")
for th, n in [(0.5, 5), (2.5, 10), (40.146, 50)]:
    lhs = np.exp(sp.gammaln(th) - sp.gammaln(th + n))
    I = integrate.quad(lambda e: e ** (th - 1) * (1 - e) ** (n - 1), 0, 1)[0] / np.exp(sp.gammaln(n))
    print(th, n, lhs, I, lhs / I - 1)
rng = np.random.default_rng(1)
# PIG0 Laplace check of single term
for t in (0.3, 1.0):
    Y = 1 / rng.gamma(1.5, 1.0, 400000)
    print("invga LT", t, np.mean(np.exp(-t * Y)), (1 + 2 * np.sqrt(t)) * np.exp(-2 * np.sqrt(t)))
print("=== C. Gibbs n=50 K=10")
n, K, a0, b0 = 50, 10, 0.01, 0.01
m = mle(n, K); print("mle", m, "se", info(m, n) ** -0.5, "EK", EK(m, n))
rng = np.random.default_rng(2026)
th = 1.0
for t in range(1, 7):
    eta = rng.beta(th, n)
    rate = b0 - np.log(eta)
    th = rng.gamma(K + a0, 1 / rate)
    print(t, round(eta, 4), round(rate, 3), round((K + a0) / rate, 3), round(th, 3))
d = gibbs(n, K, a0, b0, 100000, np.random.default_rng(7), th0=1.0, burn=1000)
s = summ(*grid_post(n, K, gamma_lp(a0, b0)))
print("gibbs", d.mean(), d.std(), np.quantile(d, [.025, .5, .975]), "iat", iat(d)[0])
print("quad", s)
# eta interpretation: E[-log eta | theta]
print("E[-log eta|theta=mle]", sp.digamma(m + n) - sp.digamma(m))
print("=== D. BvM table n=1000")
n = 1000
for K in (3, 5, 10, 20, 40, 80, 160):
    m = mle(n, K); J = VarK(m, n); se = m / np.sqrt(J)
    s = summ(*grid_post(n, K, gamma_lp(0.01, 0.01)))
    lw = m * np.exp(-1.96 / np.sqrt(J)); hw = m * np.exp(1.96 / np.sqrt(J))
    print(f"K={K:4d} mle={m:8.3f} J={J:7.2f} se={se:7.3f} wald=[{m-1.96*se:7.3f},{m+1.96*se:7.3f}] logwald=[{lw:7.3f},{hw:7.3f}] "
          f"mean={s['mean']:8.3f} med={s['median']:8.3f} cred=[{s['lo']:7.3f},{s['hi']:7.3f}] skew={s['skew']:.2f} (mean-mle)/se={(s['mean']-m)/se:.3f}")
print("=== G. two variances")
for n, th in [(200, 7.333), (1000, 4.993), (15609, 40.146), (3540, 39.355)]:
    Ek = EK(th, n); Vk = VarK(th, n)
    mpois = th * np.log(1 + n / th); mp = np.log(1 + n / th) - n / (n + th)
    print(n, th, "EK", Ek, "VarK", Vk, "ratio", Vk / Ek, "poisK", mpois, "se_cond", th / np.sqrt(Vk), "se_pois", np.sqrt(mpois) / mp,
          "approx", np.sqrt(th / np.log(n / th)), np.sqrt(th / (np.log(n / th) - 1)))
# Fisher unconditional estimates
for n, K in [(15609, 240), (3540, 178), (200, 25)]:
    f = optimize.brentq(lambda a: a * np.log(1 + n / a) - K, 0.01, 1e4)
    print("Fisher est", n, K, f, "cond", mle(n, K))
print("=== F. half Cauchy prior on sqrt theta")
hc = lambda th: -0.5 * np.log(th) - np.log1p(th)
for n, K in [(200, 25), (200, 3), (50, 10), (30, 28)]:
    print(n, K, "Ga", {k: round(float(v), 3) for k, v in summ(*grid_post(n, K, gamma_lp(0.01, 0.01))).items()})
    print(n, K, "HC", {k: round(float(v), 3) for k, v in summ(*grid_post(n, K, hc)).items()})
# three block gibbs check for 200,3 and 30,28
def gibbs3(n, K, nu, T, rng):
    th = 1.0; out = np.empty(T)
    for t in range(T):
        le = log_beta_draw(th, n, rng); xi = rng.gamma(nu, 1 / (1 + th)); th = rng.gamma(K + nu / 2, 1 / (xi - le)); out[t] = th
    return out[1000:]
for n, K in [(200, 25), (200, 3), (30, 28)]:
    d = gibbs3(n, K, 1.0, 101000, np.random.default_rng(11))
    print("gibbs3", n, K, d.mean(), np.median(d), np.quantile(d, [.025, .975]), "iat", iat(d, 2000)[0])
