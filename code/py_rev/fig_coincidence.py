"""Figure py_fig-coincidence: singletons and the coincidence probability under uniform sampling.

Left, E[A_1] against n on log log axes for sigma in {0, 0.2, 0.5, 0.7}. Right, the probability
that all abundances are distinct, with exp(-gamma) = 0.5615 dashed. The simulated values were
recovered from the original PDF by extract_fig_data.py and are read from fig_data.json.
Writes figures/py_fig-coincidence.pdf (or to SMG_FIG_DIR). Run: python3 -I fig_coincidence.py
"""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 8.5})
COL = ["#0072B2", "#D55E00", "#009E73", "#8E6BB0"]
LS = ["-", "--", "-.", (0, (1, 1.2))]
MK = ["o", "s", "^", "D"]

D = json.loads((_HERE / "fig_data.json").read_text())["coincidence"]
n = D["n"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.55))
for key, c, ls, mk in zip(["0", "0.2", "0.5", "0.7"], COL, LS, MK):
    a1.plot(n, D["A1"][key], color=c, ls=ls, lw=1.3, marker=mk, ms=4, label=rf"$\sigma={key}$")
    a2.plot(n, D["distinct"][key], color=c, ls=ls, lw=1.3, marker=mk, ms=4, label=rf"$\sigma={key}$")
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel(r"$n$"); a1.set_ylabel(r"$\mathbb{E}[A_1]$")
a1.set_ylim(0.5, 1e5)
a1.set_yticks([1, 10, 100, 1e3, 1e4])
a1.grid(True, which="both", alpha=0.25, lw=0.5)
a1.set_title("(a) singletons: bounded at $\\sigma=0$,\n$n^{\\sigma}$ beyond", fontsize=9)
a1.legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.8, handlelength=2.6, borderaxespad=0.2)
a2.axhline(np.exp(-np.euler_gamma), color="0.35", ls="--", lw=0.8)
a2.set_xscale("log"); a2.set_ylim(-0.02, 1.02)
a2.set_xlabel(r"$n$"); a2.set_ylabel("Pr(all abundances distinct)")
a2.grid(True, alpha=0.25, lw=0.5)
a2.set_title("(b) the coincidence constant\nsurvives only at $\\sigma=0$", fontsize=9)
a2.legend(frameon=False, loc="upper center", ncol=2, columnspacing=0.8, handlelength=2.6, borderaxespad=0.2)
fig.tight_layout(w_pad=1.0)
_FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(str(_FIGDIR / "py_fig-coincidence.pdf"))
plt.close(fig)
