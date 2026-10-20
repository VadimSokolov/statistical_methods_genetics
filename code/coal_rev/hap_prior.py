"""Prior averaged E[H] under the chapter's (flagless) recursion versus the correct one."""
import sys, numpy as np
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from hap_check import recursion
rng = np.random.default_rng(7)
for shape, scale, R in [(2, 4, 6000), (4, 25, 3000)]:
    th = rng.gamma(shape, scale, R)
    f = [recursion(30, t, rng, False) for t in th]
    c = [recursion(30, t, rng, True) for t in th]
    print(shape, scale, "chapter rec", np.mean(f), "+-", np.std(f)/np.sqrt(R), "correct", np.mean(c), "+-", np.std(c)/np.sqrt(R))
