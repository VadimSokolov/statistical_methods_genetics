"""Fisher information about theta in K_n (Ewens), in S, and in the levelwise vector (S_2..S_n)."""
import sys, numpy as np
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from coalsim import *

def I_K(n, th): return sum((k-1)/(th+k-1)**2 for k in range(2, n+1))/th
def I_vec(n, th): return sum(1/(th+k-1) for k in range(2, n+1))/th
def I_S(n, th, smax=4000, h=1e-5):
    p = pmf_S(n, th, smax); pp = pmf_S(n, th+h, smax); pm = pmf_S(n, th-h, smax)
    d = (pp-pm)/(2*h); m = p > 1e-300
    return np.sum(d[m]**2/p[m]), p.sum()
def bound(n, th): return a(n)**2/(th*a(n)+th**2*a(n, 2))
for n in [4, 30, 500]:
    for th in [1, 5]:
        iS, mass = I_S(n, th, smax=2000 if n < 500 else 3000)
        print(n, th, "I_K %.4f I_S %.4f I_vec %.4f  1/Var(theta_W) %.4f  log(n)/theta %.4f mass %.9f" %
              (I_K(n, th), iS, I_vec(n, th), bound(n, th), np.log(n)/th, mass))
