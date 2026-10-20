import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np, sys
from piglib import *
from scipy import special as sp
np.set_printoptions(precision=4, suppress=True)
# regimes table
for n, K in [(200, 19), (200, 4), (30, 28)]:
    m = mle(n, K)
    s = summ(*grid_post(n, K, gamma_lp(0.01, 0.01)))
    print("regime", n, K, round(m, 3), {k: round(float(v), 3) for k, v in s.items()})
# n=1000,K=27
s = summ(*grid_post(1000, 27, gamma_lp(0.01, 0.01))); m = mle(1000, 27)
print("1000/27", m, info(m, 1000) ** -0.5, s)
print("EK(5,1000)", EK(5, 1000))
# Williams
n, K = 15609, 240
m = mle(n, K); print("Williams mle", m, info(m, n) ** -0.5)
for lab, lp in [("Ga.01", gamma_lp(0.01, 0.01)), ("Ga1e-3", gamma_lp(1e-3, 1e-3)), ("Ga11", gamma_lp(1, 1)),
                ("Ga2,.05", gamma_lp(2, 0.05)), ("Ga4,.1", gamma_lp(4, 0.1)),
                ("PIG K0", pigprior_lp(10, 3)), ("PIG K0-1", pigprior_lp(10, 3, -1))]:
    print(lab, {k: round(float(v), 3) for k, v in summ(*grid_post(n, K, lp)).items()})
# 1933
m = mle(3540, 178); print("1933 mle", m, "pred", EK(m, 3540 + 12069) - EK(m, 3540))
s = summ(*grid_post(3540, 178, gamma_lp(0.01, 0.01))); print("1933 post", s)
# n=30 K=28 PIG prior
for se in (0, -1):
    print("30/28 PIG", se, summ(*grid_post(30, 28, pigprior_lp(10, 3, se))))
    print("prior median", se, summ(*grid_post(0, 0, pigprior_lp(10, 3, se), lo=-20, hi=20))["median"])
# shakespeare
print("uni mle", mle(500, 240))
for se in (0, -1):
    for n, K in [(500, 240), (100, 100), (400, 396)]:
        print("shak", se, n, K, {k: round(float(v), 2) for k, v in summ(*grid_post(n, K, pigprior_lp(10, 3, se), hi=25)).items()})
# flat prior running mean, n=20
n = 20
for K in (15, 16, 17, 18):
    phi = np.linspace(-15, 12, 400001); th = np.exp(phi)
    lp = loglik(th, n, K) + phi
    # normalising: for K=18 proper; use constant normaliser from full integral to large cutoff
    w = np.exp(lp - lp.max())
    Z = np.trapezoid(w, phi)
    cm = np.cumsum(th * w) * (phi[1] - phi[0]) / Z
    print("flat K", K, [round(float(np.interp(c, phi, cm)), 1) for c in (6, 8, 10, 12)])
# n=200, k=25
m = mle(200, 25); se_ = info(m, 200) ** -0.5
s = summ(*grid_post(200, 25, gamma_lp(0.01, 0.01)))
print("200/25", m, se_, m - 1.96 * se_, m + 1.96 * se_, s, (s['hi'] - s['median']) / (s['median'] - s['lo']))
print("unseen Williams", 40.146 / (15609 + 40.146))
