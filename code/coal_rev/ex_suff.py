"""(a) n=3, S=1: P(doubleton | S=1) = (2+theta)/(2(3+2theta)), checked by simulation.
(b) n=4, xi=(2,0,0): likelihoods of 'both singletons on one tip' (H=2) and 'on two tips' (H=3)."""
import sys, numpy as np
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from coalsim import *
from scipy import integrate

rng = np.random.default_rng(3)
for th in [0.5, 2.0, 8.0]:
    exact = (2 + th) / (2 * (3 + 2 * th))
    hits = tot = 0
    for _ in range(200000):
        o = simulate(3, th, rng)
        if o["S"] == 1:
            tot += 1; hits += o["xi"][1]
    print("n=3 theta", th, "exact", round(exact, 4), "sim", round(hits / tot, 4), "from", tot, "se", round(np.sqrt(exact*(1-exact)/tot), 4))

# n=4 tilted moments. T_k ~ Exp(lam_k), tilt exp(-theta k T_k / 2): mu_k = k(k-1+theta)/2
def tilted(theta):
    lam = {k: k*(k-1)/2 for k in (2, 3, 4)}
    mu = {k: k*(k-1+theta)/2 for k in (2, 3, 4)}
    m0 = {k: lam[k]/mu[k] for k in mu}
    m1 = {k: lam[k]/mu[k]**2 for k in mu}
    m2 = {k: 2*lam[k]/mu[k]**3 for k in mu}
    return m0, m1, m2

def E_poly(theta, coeffs):
    """E[exp(-theta L/2) * sum c * T_i T_j] for quadratic form given as dict {(i,j): c}, i,j in {2,3,4} (i<=j)."""
    m0, m1, m2 = tilted(theta)
    tot = 0.0
    for (i, j), c in coeffs.items():
        val = 1.0
        for k in (2, 3, 4):
            if i == j == k: val *= m2[k]
            elif k in (i, j): val *= m1[k]
            else: val *= m0[k]
        tot += c * val
    return tot

def quad(vecs):
    """vecs: list of external branch lengths as dicts {k: coeff}. Return (sum e^2/2, sum_{i<j} e_i e_j) as quadratic dicts."""
    def mult(u, v):
        d = {}
        for a_, ca in u.items():
            for b_, cb in v.items():
                key = (min(a_, b_), max(a_, b_)); d[key] = d.get(key, 0) + ca*cb
        return d
    def add(d, e, s=1.0):
        for k_, v in e.items(): d[k_] = d.get(k_, 0) + s*v
    same, diff = {}, {}
    for i, u in enumerate(vecs):
        add(same, mult(u, u), 0.5)
        for v in vecs[i+1:]:
            add(diff, mult(u, v))
    return same, diff

cat = [{4: 1}, {4: 1}, {4: 1, 3: 1}, {4: 1, 3: 1, 2: 1}]   # caterpillar, prob 2/3
bal = [{4: 1}, {4: 1}, {4: 1, 3: 1}, {4: 1, 3: 1}]          # balanced, prob 1/3
sc, dc = quad(cat); sb, db = quad(bal)

def likA(th):  # both singletons on one tip (unlabelled), up to theta free constants
    return (th/2)**2 * (2/3*E_poly(th, sc) + 1/3*E_poly(th, sb))
def likB(th):
    return (th/2)**2 * (2/3*E_poly(th, dc) + 1/3*E_poly(th, db))

for th in [0.1, 0.5, 1, 2, 4, 8]:
    print("theta", th, "P(A|xi=(2,0,0)) =", round(likA(th)/(likA(th)+likB(th)), 4))
# limit theta->0
print("theta->0 ratio A/(A+B)", likA(1e-8)/(likA(1e-8)+likB(1e-8)))
# simulation check at theta=2: among S=2 with xi1=2, fraction with H=2
for th in [1.0, 4.0]:
    a_ = b_ = 0
    for _ in range(300000):
        o = simulate(4, th, rng)
        if o["S"] == 2 and o["xi"][0] == 2:
            if o["H"] == 2: a_ += 1
            else: b_ += 1
    print("sim theta", th, "P(H=2 | xi=(2,0,0))", round(a_/(a_+b_), 4), "n", a_+b_, "exact", round(likA(th)/(likA(th)+likB(th)), 4))

# MLEs and posteriors under Ga(2, scale 1) prior
from scipy.optimize import minimize_scalar
for name, f in [("A (H=2)", likA), ("B (H=3)", likB)]:
    r = minimize_scalar(lambda t: -np.log(f(t)), bounds=(1e-3, 50), method="bounded")
    Z = integrate.quad(lambda t: f(t)*t*np.exp(-t), 0, np.inf)[0]
    M = integrate.quad(lambda t: t*f(t)*t*np.exp(-t), 0, np.inf)[0]
    print(name, "MLE", round(r.x, 4), "posterior mean Ga(2,1)", round(M/Z, 4))
# MLE of theta from S=2 alone at n=4
r = minimize_scalar(lambda t: -np.log(pmf_S(4, t, 5)[2]), bounds=(1e-3, 50), method="bounded")
Z = integrate.quad(lambda t: pmf_S(4, t, 5)[2]*t*np.exp(-t), 0, np.inf)[0]
M = integrate.quad(lambda t: t*pmf_S(4, t, 5)[2]*t*np.exp(-t), 0, np.inf)[0]
print("S=2 only: MLE", round(r.x, 4), "posterior mean", round(M/Z, 4))
# xi=(2,0,0) likelihood is likA+likB (up to constants consistent?)  check: P(xi=(2,0,0)) = E[e^{-thL/2}(th L1/2)^2/2]
print("check additivity:", [(round((likA(t)+likB(t)), 6)) for t in (1, 2)])
r = minimize_scalar(lambda t: -np.log(likA(t)+likB(t)), bounds=(1e-3, 50), method="bounded")
Z = integrate.quad(lambda t: (likA(t)+likB(t))*t*np.exp(-t), 0, np.inf)[0]
M = integrate.quad(lambda t: t*(likA(t)+likB(t))*t*np.exp(-t), 0, np.inf)[0]
print("spectrum xi=(2,0,0): MLE", round(r.x, 4), "posterior mean", round(M/Z, 4))
