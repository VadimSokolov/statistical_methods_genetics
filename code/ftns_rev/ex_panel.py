import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import io,contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from ex_horseshoe import post, y
import numpy as np
for tau in [1.018, 0.2]:
    E=[];T=0;P=0
    for v in y:
        Z,Ek,Ek2=post(v,tau); Ed=(1-Ek)*v; Ed2=(1-Ek)+v*v*(1-2*Ek+Ek2); E.append((v,Ek,Ed,Ed2)); T+=Ed2; P+=Ed**2
    print('tau',tau, 'total',round(T,2),'plug',round(P,2))
    for r in E: print('  %5.1f %.3f %6.3f %6.3f'%r)
print('sum y2',(y*y).sum(),'minus L',(y*y).sum()-len(y))
