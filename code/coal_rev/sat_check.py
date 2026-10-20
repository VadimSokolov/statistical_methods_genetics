"""Check exact posterior coverage under Ga(4,25) with S drawn from the geometric law and from the simulator test set."""
import sys, numpy as np
from scipy import stats
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
n=30; shape, scale = 4.0, 25.0
te = np.load(str(_HERE / "sat_test.npz"))
rng = np.random.default_rng(9)
M = 12000
th = rng.gamma(shape, scale, M); s = np.zeros(M, int)
for k in range(2, n+1): s += rng.geometric((k-1)/(k-1+th)) - 1
for G in [(500, 7000), (900, 30000)]:
    grid = np.linspace(1e-3, G[0], G[1]); smax = max(s.max(), te["xi"].sum(1).max())+5
    P = np.zeros((smax+1, grid.size)); P[0] = 1
    for k in range(2, n+1):
        p = (k-1)/(k-1+grid); q = 1-p
        new = np.empty_like(P); new[0] = p*P[0]
        for j in range(1, smax+1): new[j] = q*new[j-1]+p*P[j]
        P = new
    post = P*stats.gamma.pdf(grid, shape, scale=scale); cdf = np.cumsum(post, 1); cdf /= cdf[:, -1:]
    for name, T, SS in [("geometric draws", th, s), ("simulator test", te["th"], te["xi"].sum(1))]:
        u = np.array([np.interp(t, grid, cdf[x]) for t, x in zip(T, SS)])
        print(G, name, "cov90", np.mean((u > .05) & (u < .95)).round(4), "cov50", np.mean((u > .25) & (u < .75)).round(4),
              "chi2", (((np.histogram(u, 10, (0, 1))[0]-len(u)/10)**2)/(len(u)/10)).sum().round(1))
