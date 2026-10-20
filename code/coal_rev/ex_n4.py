"""Worked examples at n = 4: exact pmf of S, coupling K' = 1 + #{k: S_k >= 1}, haplotype count H."""
import sys, numpy as np
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from coalsim import *
from fractions import Fraction

n = 4
print("geometric parameters at theta=1:", [Fraction(k-1, k) for k in range(2, 5)])
for th in [1]:
    print("theta", th)
    for s in range(0, 8):
        f = pmf_S_frac(n, th, s)
        print(s, f, float(f), pmf_S_tavare(n, th, s), pmf_S(n, th, 10)[s])
print("P(S<=7) at theta=1", float(sum(pmf_S_frac(4, 1, s) for s in range(8))))

rng = np.random.default_rng(20261017)
R = 200000
th = 1.0
S = np.zeros(R, int); Kp = np.zeros(R, int); H = np.zeros(R, int); xi = np.zeros((R, 3), int)
for r in range(R):
    o = simulate(n, th, rng)
    S[r] = o["S"]; H[r] = o["H"]; xi[r] = o["xi"]
    Kp[r] = 1 + sum(1 for k in o["Sk"] if o["Sk"][k] >= 1)
ex = pmf_S(n, th, 20)
print("sim vs exact pmf, theta=1")
for s in range(8):
    p = np.mean(S == s); se = np.sqrt(ex[s]*(1-ex[s])/R)
    print(s, round(p, 4), round(ex[s], 4), "z=", round((p-ex[s])/se, 2))
obs = np.array([np.sum(S == s) for s in range(8)] + [np.sum(S >= 8)])
expc = R*np.array(list(ex[:8]) + [1-ex[:8].sum()])
print("chi2 (8 df)", ((obs-expc)**2/expc).sum())
print("mean S", S.mean(), "exact", th*a(n), "var", S.var(), th*a(n)+th**2*a(n, 2))
# Ewens law of K at theta=1, n=4: |s(4,k)|/4!
ew = {1: 6/24, 2: 11/24, 3: 6/24, 4: 1/24}
print("K' and H vs Ewens")
for k in range(1, 5):
    print(k, round(np.mean(Kp == k), 4), round(np.mean(H == k), 4), ew[k])
print("joint table P(K'=i, H=j)")
for i in range(1, 5):
    print(i, [round(np.mean((Kp == i) & (H == j)), 4) for j in range(1, 5)])
print("P(K' != H)", np.mean(Kp != H))
print("E[S | K'=k]", [round(S[Kp == k].mean(), 3) for k in range(1, 5)])
print("E[S | H=k]", [round(S[H == k].mean(), 3) for k in range(1, 5)])
# exact E[S | K'=k] : S_k | S_k>=1 has mean 1/p_k  (memoryless: 1 + Geom)
from itertools import product
for k in range(1, 5):
    tot = 0; pr = 0
    for marks in product([0, 1], repeat=3):
        if 1 + sum(marks) != k: continue
        w = 1; m = 0
        for kk, b in zip([2, 3, 4], marks):
            q = th/(th+kk-1)
            w *= q if b else 1-q
            m += (1/(1-q)) if b else 0
        tot += w*m; pr += w
    print("exact E[S|K'=%d]" % k, tot/pr, "P", pr)
