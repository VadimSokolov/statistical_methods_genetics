import numpy as np, sys
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import io,contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from ex_horseshoe import post
from scipy.optimize import brentq
for tau in [0.15,0.07,0.04,0.02, 0.5,0.18,0.05]:
    f=lambda y: (1-post(y,tau)[1])-0.5
    print(tau, round(brentq(f,0.01,20),2))
# variant: what if 'retained fraction' is computed with tau^2 replaced, find tau' so that y=0 gives 0.051
from scipy.optimize import brentq as b
for target in [0.051,0.011,0.001]:
    t=b(lambda t:(1-post(0.0,t)[1])-target,1e-4,0.99); print('tau giving',target,t,[round(1-post(v,t)[1],3) for v in [0,2,3,4,6,8]])
