import numpy as np, sys, io, contextlib
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
with contextlib.redirect_stdout(io.StringIO()):
    from ex_bayes import logK
n=500
for s in [(331,372,302),(331,318,302),(331,316.5,302),(331,325,302),(331,340,302)]:
    K=np.exp(logK(s,n)); print(s,'K',K,'post prob H0 (equal odds)',K/(1+K))
