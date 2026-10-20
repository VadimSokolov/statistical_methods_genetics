import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np
from ftns_rev_lib import fit, simulate
rng=np.random.default_rng(2024)
for frac,sd in [(0.10,3.0),(0.05,3.0),(0.02,2.5),(0.01,4.0)]:
    d,y=simulate(rng,5000,frac,sd); P=fit(y); sq=np.sort(P['Ed']**2)[::-1]
    print(frac, round(P['tau'],3), 'top5 of (Ed)^2', round(sq[:250].sum()/sq.sum(),3), 'share of truth in top 5% true', round(np.sort(d*d)[::-1][:250].sum()/(d*d).sum(),3))
