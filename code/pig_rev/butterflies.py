"""Good Turing and Bayesian log series smoothing for Corbet's Malayan butterflies,
frequencies of frequencies r = 1..15 (Fisher, Corbet and Williams 1943). python -I butterflies.py"""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np, sys, json
from scipy import special as sp, optimize
a = np.array([118, 74, 44, 24, 29, 22, 20, 19, 20, 15, 12, 14, 6, 12, 6], float)
r = np.arange(1, 16)
print("species r<=15", a.sum(), "specimens", (r * a).sum())
gt_star = (r[:-1] + 1) * a[1:] / a[:-1]
print("raw r*", np.round(gt_star, 3))
print("Good Toulmin t=1", np.sum((-1) ** (r + 1) * a), "sd", np.sqrt(a.sum()))
def nll(p):
    th, x = np.exp(p[0]), 1 / (1 + np.exp(-p[1]))
    mu = th * x ** r / r
    return -np.sum(a * np.log(mu) - mu)
res = optimize.minimize(nll, [np.log(100), 3.0], method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000))
th, x = np.exp(res.x[0]), 1 / (1 + np.exp(-res.x[1]))
print("MLE theta", th, "x", x, "fitted", np.round(th * x ** r / r, 2))
print("chi2", np.sum((a - th * x ** r / r) ** 2 / (th * x ** r / r)))
# posterior on grid: prior flat in log theta and uniform in x
TH = np.exp(np.linspace(np.log(60), np.log(250), 600)); X = np.linspace(0.85, 1.0, 1501)[:-1]
T, XX = np.meshgrid(TH, X, indexing="ij")
lp = np.zeros_like(T)
for ri, ai in zip(r, a):
    mu = T * XX ** ri / ri
    lp += ai * np.log(mu) - mu
lp += 0  # flat in log theta: grid is uniform in log theta already
w = np.exp(lp - lp.max()); w /= w.sum()
def q(f):
    v = f.ravel(); p = w.ravel(); o = np.argsort(v); c = np.cumsum(p[o])
    return np.sum(v * p), np.interp([0.025, 0.975], c, v[o])
print("post theta", q(T), "post x", q(XX))
for rr in (1, 2, 3, 5, 10):
    print("r", rr, "smoothed r*", q(rr * XX))
print("new species t=1 theta log(1+x)", q(T * np.log1p(XX)))
print("check Good Toulmin under fitted log series r<=15", np.sum((-1) ** (r + 1) * th * x ** r / r), th * np.log1p(x))
