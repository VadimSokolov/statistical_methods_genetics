"""Figure py_fig-index: only the index of G matters.

The singleton fraction A_1/K_n against n for three weights of index rho = 2 (G(r) = r^2,
r^2/(1 - 0.5 log r), r^2 (3 - 2 r)) at sigma = 1/2, theta = 1; all converge to sigma_* = 1/3
(dashed). The simulated values were recovered from the original PDF by extract_fig_data.py and
are read from fig_data.json. Writes figures/py_fig-index.pdf (or to SMG_FIG_DIR).
Run: python3 -I fig_index.py
"""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 8.5})
COL = ["#0072B2", "#D55E00", "#009E73"]
LS = ["-", "-.", (0, (1, 1.2))]
MK = ["o", "s", "^"]

D = json.loads((_HERE / "fig_data.json").read_text())["index"]
rows = [("pure", r"$G(r)=r^2$"), ("log", r"$G(r)=r^2/(1-0.5\,\log r)$"), ("poly", r"$G(r)=r^2(3-2r)$")]
fig, ax = plt.subplots(figsize=(3.8, 2.7))
ax.axhline(1 / 3, color="k", ls="--", lw=1.0, label=r"$\sigma_*=1/3$")
for (k, lab), c, ls, mk in zip(rows, COL, LS, MK):
    ax.plot(D["n"], D[k], color=c, ls=ls, lw=1.3, marker=mk, ms=4, label=lab)
ax.set_xscale("log"); ax.set_ylim(0.2, 0.6)
ax.set_xlabel(r"$n$"); ax.set_ylabel(r"$A_1/K_n$")
ax.grid(True, alpha=0.25, lw=0.5)
ax.set_title(r"Three weights of the same index $\rho=2$," "\nthree different shapes", fontsize=9)
ax.legend(frameon=False, loc="upper right", handlelength=2.6, borderaxespad=0.2)
fig.tight_layout()
_FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(str(_FIGDIR / "py_fig-index.pdf"))
plt.close(fig)
