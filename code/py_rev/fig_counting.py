"""Figure py_fig-counting: counting functions E[N(x)] of the induced frequencies against 1/x.

Four (sigma, alpha) pairs over four decades in x on log log axes; the slope is sigma_*. The
simulated points were recovered from the original PDF by extract_fig_data.py and are read
from fig_data.json; the fitted and predicted slopes in the legend are those of the original.
Writes figures/py_fig-counting.pdf (or to SMG_FIG_DIR). Run: python3 -I fig_counting.py
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
COL = ["#0072B2", "#D55E00", "#009E73", "#8E6BB0"]
LS = ["-", "--", "-.", (0, (1, 1.2))]
MK = ["o", "s", "^", "D"]

D = json.loads((_HERE / "fig_data.json").read_text())["counting"]
rows = [("s0.5_a1", r"$\sigma=0.5,\ \alpha=1$: fitted 0.489, predicted 0.500"),
        ("s0.5_a2", r"$\sigma=0.5,\ \alpha=2$: fitted 0.345, predicted 0.333"),
        ("s0.5_a0.5", r"$\sigma=0.5,\ \alpha=0.5$: fitted 0.542, predicted 0.667"),
        ("s0.3_a2", r"$\sigma=0.3,\ \alpha=2$: fitted 0.208, predicted 0.176")]
fig, ax = plt.subplots(figsize=(5.4, 3.0))
for (key, lab), c, ls, mk in zip(rows, COL, LS, MK):
    ax.plot(D[key]["x"], D[key]["y"], color=c, ls=ls, lw=1.3, marker=mk, ms=4, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$1/x$"); ax.set_ylabel(r"$\mathbb{E}[N(x)]$")
ax.set_ylim(top=1.2e6)
ax.grid(True, which="both", alpha=0.25, lw=0.5)
ax.set_title(r"Counting function: slope is $\sigma_*$", fontsize=9)
ax.legend(frameon=False, loc="upper left", handlelength=2.8, borderaxespad=0.2)
fig.tight_layout()
_FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(str(_FIGDIR / "py_fig-counting.pdf"))
plt.close(fig)
