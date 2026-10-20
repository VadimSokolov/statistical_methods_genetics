import numpy as np, sys
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from ex_horseshoe import post
rng=np.random.default_rng(1)
lam=np.abs(rng.standard_cauchy(2_000_000))
for tau in [0.5]:
    k=1/(1+tau**2*lam**2)
    for y in [0,2,3]:
        w=np.sqrt(k)*np.exp(-k*y*y/2); print('MC',tau,y,1-(w*k).sum()/w.sum())
for t2 in [0.25,0.18**2,0.05**2]:
    print('tau',t2,[round(1-post(v,t2)[1],3) for v in [0,2,3,4,6,8]])
