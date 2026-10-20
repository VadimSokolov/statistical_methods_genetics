"""Figure ftns_f2_xor: the exclusive or system (Chapter ftns, Figure ftns:fig:xor).

Left: mean fitness over generations 0 to 15 against Fisher's prediction, which is flat because
the additive genetic variance is zero in every generation. Right: the disequilibrium D and the
allele frequency deviation p_A - 0.5, which stays at zero. The plotted values were recovered
from the original PDF by extract_ftns_figdata.py and are read from ftns_figdata.json.
Redrawn at the house width of 5.4 in; series differ by marker and line style as well as hue.
"""
import os as _os, sys as _sys, json
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
BLUE, RED, GREY = "#1f4e79", "#c0392b", "#8a9aa5"

d = json.load(open(_HERE / "ftns_figdata.json"))["f2"]
g = np.array(d["gen"]); w = np.array(d["wbar"]); f = d["fisher"]

fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.3))
a = ax[0]
a.fill_between(g, w, f, color=RED, alpha=0.12, lw=0)
a.plot(g, w, "-o", color=BLUE, lw=1.4, ms=3.5, label=r"$\bar w$ (true)")
a.plot(g, np.full_like(g, f, dtype=float), "--", color=RED, lw=1.4, label="Fisher's prediction")
a.set_xlabel("generation"); a.set_ylabel(r"$\bar w$")
a.set_ylim(0.85, 1.10)
a.set_title(r"$\sigma^2_A=0$ in every generation", fontsize=9, color=BLUE)
a.legend(frameon=False, loc="center right")

a = ax[1]
a.axhline(0, color="k", lw=0.7, ls=":")
a.plot(g, d["D"], "-o", color=RED, lw=1.4, ms=3.5, label="disequilibrium $D$\n(transverse)")
a.plot(g, d["pA_minus_half"], "-s", color=GREY, lw=1.4, ms=3.5, mfc="white",
       label="$p_A-0.5$\n(tangential, $\\equiv0$)")
a.set_xlabel("generation")
a.set_ylim(-0.21, 0.01)
a.set_title("all motion is transverse", fontsize=9, color=RED)
a.legend(frameon=False, loc="lower right", handlelength=1.6, labelspacing=0.6)
fig.tight_layout(w_pad=1.0)
fig.savefig(str(_FIGDIR / "ftns_f2_xor.pdf"))
print("wrote", _FIGDIR / "ftns_f2_xor.pdf")
