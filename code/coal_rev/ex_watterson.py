"""Simulated sample, n = 11, theta = 4: Watterson's estimator, variance, exact likelihood, exact CI, posterior, pi, Tajima's D."""
import sys, numpy as np
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from coalsim import *
from scipy.optimize import brentq, minimize_scalar

n, th0 = 11, 4.0
rng = np.random.default_rng(1983)
o = simulate(n, th0, rng, return_sites=True)
sites = sorted(o["sites"], key=lambda b: (-bin(b).count("1"), b))
S = len(sites)
X = np.array([[(b >> leaf) & 1 for b in sites] for leaf in range(n)])
# order rows lexicographically for display
order = sorted(range(n), key=lambda r: tuple(-X[r]))
X = X[order]
for r in X: print("".join(map(str, r)))
xi = o["xi"]; H = o["H"]
print("S", S, "xi", list(xi), "H", H, "Sk", o["Sk"])
a1, a2 = a(n), a(n, 2)
thW = S/a1
varhat = S/a1**2 + a2*S*(S-1)/(a1**2*(a1**2+a2))
print("a1 %.6f a2 %.6f thetaW %.4f varhat %.4f se %.4f" % (a1, a2, thW, varhat, np.sqrt(varhat)))
print("plug in variance thW/a1 + thW^2 a2/a1^2 =", thW/a1 + thW**2*a2/a1**2)
pi = sum(i*(n-i)*xi[i-1] for i in range(1, n))/(n*(n-1)/2)
b1 = (n+1)/(3*(n-1)); b2 = 2*(n*n+n+3)/(9*n*(n-1))
c1 = b1-1/a1; c2 = b2-(n+2)/(a1*n)+a2/a1**2
e1 = c1/a1; e2 = c2/(a1**2+a2)
D = (pi-S/a1)/np.sqrt(e1*S+e2*S*(S-1))
print("pi %.4f TajimaD %.4f" % (pi, D))
# exact likelihood
L = lambda t: pmf_S(n, t, S+1)[S]
r = minimize_scalar(lambda t: -np.log(L(t)), bounds=(1e-3, 100), method="bounded")
print("MLE from S exact", r.x)
cdf = lambda t, s: pmf_S(n, t, s+1)[:s+1].sum()
# exact 90% CI: lower solves P(S>=S_obs)=0.05, upper solves P(S<=S_obs)=0.05
lo = brentq(lambda t: (1-cdf(t, S-1)) - 0.05, 1e-4, 100)
hi = brentq(lambda t: cdf(t, S) - 0.05, 1e-4, 200)
print("exact 90%% CI (%.3f, %.3f)" % (lo, hi))
print("normal approx CI", thW-1.645*np.sqrt(varhat), thW+1.645*np.sqrt(varhat))
# posterior under exponential prior mean 10 and under log uniform? use Ga(1, scale 10)
g = np.linspace(1e-4, 60, 60001)
like = np.array([L(t) for t in g[::10]])
gg = g[::10]
for name, prior in [("Exp(mean 10)", np.exp(-gg/10)), ("1/theta", 1/gg)]:
    post = like*prior; post /= np.trapezoid(post, gg)
    c = np.cumsum(post)*(gg[1]-gg[0])
    q = [gg[np.searchsorted(c, u)] for u in (0.05, 0.5, 0.95)]
    print(name, "post mean %.3f" % np.trapezoid(gg*post, gg), "quantiles", np.round(q, 3))
# repeat sampling check of varhat unbiasedness and coverage of exact CI at theta0
R = 20000
Ss = np.array([simulate(n, th0, rng)["S"] for _ in range(R)])
print("MC: mean thetaW", np.mean(Ss/a1), "var", np.var(Ss/a1), "theory var", th0/a1+th0**2*a2/a1**2,
      "mean varhat", np.mean(Ss/a1**2 + a2*Ss*(Ss-1)/(a1**2*(a1**2+a2))))
