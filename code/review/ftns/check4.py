import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import sys,io,contextlib,numpy as np
sys.path.insert(0,str(_ROOT / "code" / "ftns_rev"))
with contextlib.redirect_stdout(io.StringIO()):
    from ex_bayes import logK
for th in [(0.45,0.60,0.35),(0.45,0.42,0.35),(0.45,0.40,0.35)]:
    print(th,[ '%.3g'%np.exp(logK([n*t for t in th],n)) for n in [25,50,100,200,400,1000,4000]])
