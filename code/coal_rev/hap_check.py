"""Check E[H] = Ewens E[K_n]; compare the chapter's merge recursion (without ancestral type flag) with the correct one."""
import sys, numpy as np
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from coalsim import *
from scipy import integrate, stats

n = 30
for shape, scale in [(2, 4), (4, 25)]:
    f = lambda t: ewens_EK(n, t) * stats.gamma.pdf(t, shape, scale=scale)
    print("prior Ga(%g,%g): E[H] exact =" % (shape, scale), integrate.quad(f, 0, np.inf, limit=200)[0],
          " E[S] =", shape*scale*a(n))
print("E[H] theta=0.01:", ewens_EK(n, 0.01), " theta=1000:", ewens_EK(n, 1000))


def recursion(n, theta, rng, correct):
    # nodes: (c, anc_present); leaves c=1, anc=1
    nodes = [(1, 1)] * n
    times = [0.0] * n  # time each lineage was born (node time)
    t = 0.0
    for k in range(n, 1, -1):
        t += rng.exponential(2.0 / (k * (k - 1)))
        i, j = rng.choice(k, size=2, replace=False)
        (cA, aA), (cB, aB) = nodes[i], nodes[j]
        freeA = rng.random() < np.exp(-theta * (t - times[i]) / 2)
        freeB = rng.random() < np.exp(-theta * (t - times[j]) / 2)
        if correct:
            c = cA + cB - (aA and aB and freeA and freeB)
            anc = (aA and freeA) or (aB and freeB)
        else:
            c = cA + cB - (freeA and freeB)
            anc = 1
        nodes = [x for s, x in enumerate(nodes) if s not in (i, j)] + [(c, anc)]
        times = [x for s, x in enumerate(times) if s not in (i, j)] + [t]
    return nodes[0][0]

rng = np.random.default_rng(1)
for th in [2.0, 8.0]:
    R = 3000
    hs = [simulate(n, th, rng)["H"] for _ in range(R)]
    rc = [recursion(n, th, rng, True) for _ in range(R)]
    rf = [recursion(n, th, rng, False) for _ in range(R)]
    print("theta", th, "Ewens", round(ewens_EK(n, th), 3), "direct H", np.mean(hs), "correct rec", np.mean(rc),
          "chapter rec", np.mean(rf), "min chapter rec", np.min(rf))
