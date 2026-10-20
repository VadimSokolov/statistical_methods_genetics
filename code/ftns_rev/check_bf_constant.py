"""Check of the constant c in K ~ c sqrt(n) (Section ftns:sec:ockham and the exercise on zygote numbers).

Reproduces: the d=0 column of Table ftns:tab:bf (exact binomial Bayes factor at expected null data,
theta=(0.45,0.40,0.35), windowed grid quadrature as in ex_bayes.py), the ratio K/sqrt(n) at every
tabulated n, the Laplace limit c = [2 pi (v12 + (v11+v22)/4)]^(-1/2) with v_g = theta_g(1-theta_g),
and the derived numbers (L/(c log 2))^2 for L = 1e3, 1e4, c sqrt(1e5) and (100/c)^2.

Run: python3 -I code/ftns_rev/check_bf_constant.py   (output cached in check_bf_constant.txt)
"""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np
from scipy.special import betaln


def logK(s, n, G=1600):
    """log Bayes factor H0 (additive) over H1, same algorithm as ex_bayes.py."""
    s11, s12, s22 = s
    l1 = sum(betaln(si + 1, n - si + 1) for si in s)
    def ll(a, b):
        t = (a + b) / 2
        return (s11*np.log(a) + (n-s11)*np.log1p(-a) + s22*np.log(b) + (n-s22)*np.log1p(-b)
                + s12*np.log(t) + (n-s12)*np.log1p(-t))
    g = np.linspace(1e-6, 1-1e-6, 801); AA, BB = np.meshgrid(g, g, indexing='ij'); L = ll(AA, BB)
    i, j = np.unravel_index(np.argmax(L), L.shape); a0, b0 = g[i], g[j]
    w = 8/np.sqrt(n) + 0.02
    ga = np.linspace(max(1e-9, a0-w), min(1-1e-9, a0+w), G); gb = np.linspace(max(1e-9, b0-w), min(1-1e-9, b0+w), G)
    AA, BB = np.meshgrid(ga, gb, indexing='ij'); L = ll(AA, BB); M = L.max()
    I = np.trapezoid(np.trapezoid(np.exp(L-M), gb, axis=1), ga)
    return M + np.log(I) - l1


th = np.array([0.45, 0.40, 0.35])
v = th*(1-th)
c_lap = 1/np.sqrt(2*np.pi*(v[1] + (v[0]+v[2])/4))
print(f"Laplace limit c = {c_lap:.4f}")
for n in [25, 50, 100, 200, 400, 1000, 4000, 20000, 100000]:
    K = np.exp(logK(n*th, n))
    print(f"n={n:6d}  K={K:8.2f}  K/sqrt(n)={K/np.sqrt(n):.4f}")
c = 0.67
for L in [1000, 10000]:
    print(f"L={L}: (L/(c log2))^2 = {(L/(c*np.log(2)))**2:.3e}   with c={c_lap:.4f}: {(L/(c_lap*np.log(2)))**2:.3e}")
print(f"c sqrt(1e5) = {c*np.sqrt(1e5):.1f}; (100/c)^2 = {(100/c)**2:.0f}")
