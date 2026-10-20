"""Randomised PIT of simulated S given theta under the exact geometric law, for both simulated data sets."""
import sys, numpy as np
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from coalsim import pmf_S
rng = np.random.default_rng(0)
for f in ["sat_train.npz", "sat_test.npz", "hap_train.npz"]:
    d = np.load(str(_HERE / f)); th = d["th"][:20000]; S = d["xi"].sum(1)[:20000]
    u = np.empty(len(th))
    for i, (t, s) in enumerate(zip(th, S)):
        p = pmf_S(30, t, s+1)
        u[i] = p[:s].sum() + rng.random()*p[s]
    c = np.histogram(u, 10, (0, 1))[0]; e = len(u)/10
    print(f, "chi2", ((c-e)**2/e).sum().round(1), c, "tails <.05 >.95", np.mean(u < .05).round(4), np.mean(u > .95).round(4))
