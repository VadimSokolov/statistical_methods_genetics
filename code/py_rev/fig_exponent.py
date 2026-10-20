"""Figure py_fig-exponent: the exponent map sigma_* = sigma / {alpha (1 - sigma) + sigma}.

Left, sigma_* against alpha for sigma in {0.2, 0.4, 0.6, 0.8}, with the unweighted values as
dotted horizontal lines and alpha = 1 marked. Right, the odds (1 - sigma_*)/sigma_* =
alpha (1 - sigma)/sigma on log log axes, a line of slope one for every sigma. Closed form, no
data. Writes figures/py_fig-exponent.pdf (or to SMG_FIG_DIR). Run: python3 -I fig_exponent.py
"""
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

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 8.5})
COL = ["#0072B2", "#D55E00", "#009E73", "#8E6BB0"]
LS = ["-", "--", "-.", (0, (1, 1.2))]
MK = ["o", "s", "^", "D"]

alpha = np.logspace(-1, 1, 200)
sigmas = [0.2, 0.4, 0.6, 0.8]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.7))
for s, c, ls, mk in zip(sigmas, COL, LS, MK):
    ss = s / (alpha * (1 - s) + s)
    a1.plot(alpha, ss, color=c, ls=ls, lw=1.4, marker=mk, ms=3.5, markevery=(10, 36), label=rf"$\sigma={s}$")
    a1.axhline(s, color=c, ls=":", lw=0.7)
    a2.plot(alpha, (1 - ss) / ss, color=c, ls=ls, lw=1.4, marker=mk, ms=3.5, markevery=(10, 36))
a1.axvline(1.0, color="0.3", ls="-.", lw=0.7)
a1.set_xscale("log"); a1.set_xlabel(r"$\alpha$"); a1.set_ylabel(r"$\sigma_*$")
a1.set_title(r"(a) the exponent" "\n" r"$\sigma_*=\sigma/\{\alpha(1-\sigma)+\sigma\}$", fontsize=9)
a1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
h, l = a1.get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=4, handlelength=2.6, borderaxespad=0.2)
a2.set_xscale("log"); a2.set_yscale("log")
a2.set_xlabel(r"$\alpha$"); a2.set_ylabel(r"$(1-\sigma_*)/\sigma_*$")
a2.grid(True, which="both", alpha=0.25, lw=0.5)
a2.set_title("(b) in the odds coordinate every\ncurve is a line of slope one", fontsize=9)
fig.tight_layout(w_pad=1.0, rect=(0, 0.07, 1, 1))
_FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(str(_FIGDIR / "py_fig-exponent.pdf"))
plt.close(fig)
