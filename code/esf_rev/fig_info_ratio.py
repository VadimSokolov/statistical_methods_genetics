"""Figure: Fisher information in K_n as a fraction of the tree information E L_n / (2 theta)."""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
ns = np.unique(np.round(np.logspace(np.log10(2), 6, 200)).astype(int))
fig, ax = plt.subplots(figsize=(3.8, 2.6))
for th, ls in ((0.5, "-"), (2.0, "--"), (10.0, ":")):
    i = np.arange(0, ns.max(), dtype=float)
    cumI = np.cumsum(i / (th * (th + i) ** 2))
    j = np.arange(1, ns.max(), dtype=float)
    cumH = np.concatenate([[0.0], np.cumsum(1.0 / j)])
    r = cumI[ns - 1] / (cumH[ns - 1] / th)
    ax.plot(ns[ns >= 2], r[ns >= 2], ls, color="black", lw=1, label=r"$\theta = %g$" % th)
ax.set_xscale("log")
ax.set_ylim(0, 1)
ax.set_xlabel(r"sample size $n$")
ax.set_ylabel(r"$I_n(\theta) \,/\, \{\mathrm{E}\,L_n/(2\theta)\}$")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(str(_FIGDIR) + "/esf_fig_inforatio.pdf")
print("saved")
