"""Simulation based calibration with the exact posterior given S (n = 30, Ga(3, scale 3) prior), versus
(i) the prior reported as posterior, (ii) a Poisson gamma approximation that ignores tree variance,
(iii) that approximation after global rank recalibration. Also conditional coverage by S."""
import sys, numpy as np
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from coalsim import *
from scipy import stats

n = 30; shape, scale = 3.0, 3.0
a1, a2 = a(n), a(n, 2)
grid = np.linspace(1e-3, 80, 8000)
smax = 700
# pmf of S for every theta on the grid: recursion P_new(s) = q P_new(s-1) + p P_old(s)
P = np.zeros((smax+1, grid.size)); P[0] = 1.0
for k in range(2, n+1):
    p = (k-1)/(k-1+grid); q = 1-p
    new = np.empty_like(P); new[0] = p*P[0]
    for s in range(1, smax+1):
        new[s] = q*new[s-1] + p*P[s]
    P = new
print("min mass on grid", P.sum(0)[grid < 60].min())
prior = stats.gamma.pdf(grid, shape, scale=scale)
post = P*prior; post /= post.sum(1, keepdims=True)
cdf_exact = np.cumsum(post, 1)

def pit_exact(th, s):
    return np.array([np.interp(t, grid, cdf_exact[si]) for t, si in zip(th, s)])

rng = np.random.default_rng(11)
def draw(M):
    th = rng.gamma(shape, scale, M)
    # S | theta: sum of geometrics; numpy geometric counts trials, so subtract 1
    s = np.zeros(M, int)
    for k in range(2, n+1):
        s += rng.geometric((k-1)/(k-1+th)) - 1
    return th, s
M = 20000
th, s = draw(M); thc, sc = draw(M)
print("max S", s.max())
def chi2(u, bins=10):
    c = np.histogram(u, bins=bins, range=(0, 1))[0]; e = len(u)/bins
    return ((c-e)**2/e).sum(), c
u_exact = pit_exact(th, s)
u_prior = stats.gamma.cdf(th, shape, scale=scale)
pg = lambda t, ss: stats.gamma.cdf(t, shape+ss, scale=1/(1/scale+a1))
u_pg = pg(th, s)
crit = stats.chi2.ppf(0.95, 9)
for name, u in [("exact", u_exact), ("prior", u_prior), ("poisson gamma", u_pg)]:
    x, c = chi2(u)
    print(name, "chi2 %.1f" % x, "counts", c, "cov90", np.mean((u > 0.05) & (u < 0.95)).round(3))
print("crit", crit)
# global recalibration of poisson gamma: G = ecdf of PIT on calibration set
G = np.sort(pg(thc, sc))
u_rc = np.searchsorted(G, u_pg)/M
x, c = chi2(u_rc); print("recalibrated chi2 %.1f" % x, c, "cov90", np.mean((u_rc > 0.05) & (u_rc < 0.95)).round(3))
# conditional coverage of 90% intervals by S tercile
cuts = np.quantile(s, [1/3, 2/3])
grp = np.digitize(s, cuts)
for name, u in [("exact", u_exact), ("prior", u_prior), ("poisson gamma", u_pg), ("recalibrated", u_rc)]:
    cov = [np.mean(((u > 0.05) & (u < 0.95))[grp == g]) for g in range(3)]
    print(name, "cov by S tercile", np.round(cov, 3))
print("S tercile cut points", cuts, "group sizes", np.bincount(grp))
# mean 90% widths
def qexact(si, u): return np.interp(u, cdf_exact[si], grid)
w_exact = np.mean([qexact(si, .95)-qexact(si, .05) for si in s[:4000]])
w_pg = np.mean(stats.gamma.ppf(.95, shape+s, scale=1/(1/scale+a1)) - stats.gamma.ppf(.05, shape+s, scale=1/(1/scale+a1)))
w_prior = stats.gamma.ppf(.95, shape, scale=scale)-stats.gamma.ppf(.05, shape, scale=scale)
print("mean 90 widths exact %.3f pg %.3f prior %.3f" % (w_exact, w_pg, w_prior))
# Monte Carlo standard error of coverage at 90 in a tercile
print("MC se of coverage in tercile", np.sqrt(.9*.1/(M/3)))
np.savez(str(_HERE / "sbc.npz"), u_exact=u_exact, u_prior=u_prior, u_pg=u_pg, u_rc=u_rc)
# widths of the recalibrated Poisson gamma interval: levels G^{-1}(0.05), G^{-1}(0.95)
lo_l, hi_l = np.quantile(G, 0.05), np.quantile(G, 0.95)
w_rc = np.mean(stats.gamma.ppf(hi_l, shape+s, scale=1/(1/scale+a1)) - stats.gamma.ppf(lo_l, shape+s, scale=1/(1/scale+a1)))
print("recal levels %.4f %.4f width %.3f" % (lo_l, hi_l, w_rc))
w_exact_all = np.mean([qexact(si, .95)-qexact(si, .05) for si in s])
print("exact width all %.3f" % w_exact_all)
