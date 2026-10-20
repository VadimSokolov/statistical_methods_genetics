import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import sys; sys.path.insert(0,str(_ROOT / "code" / "pig_rev"))
import numpy as np
from piglib import *
rng=np.random.default_rng(5)
for seed in [5,6]:
    rng=np.random.default_rng(seed); th=1.0; T=200000; out=np.empty(T)
    for t in range(T):
        le=log_beta_draw(th,30,rng); xi=rng.gamma(1,1/(1+th)); th=rng.gamma(28.5,1/(xi-le)); out[t]=th
    out=out[2000:]; print(np.median(out),np.quantile(out,[.025,.975]),iat(np.log(out),maxlag=2000)[0],iat(out,maxlag=2000)[0])
