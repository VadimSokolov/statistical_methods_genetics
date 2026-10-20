"""Figure ftns_f6_bf: the asymmetry of a sharp null (Chapter ftns, Figure ftns:fig:bf).

Jeffreys' Bayes factor K for kappa = 1 against the number n of zygotes per genotype, for a
visible dominance effect d = 0.20, a small deviation d = 0.02 and the null d = 0, with a
reference line proportional to sqrt(n) and a dotted line at K = 1. Under the null K reaches
211 at n = 100000. The plotted values were recovered from the original PDF by
extract_ftns_figdata.py and are read from ftns_figdata.json (the d = 0.20 curve leaves the
plotted range after n = 1000). Redrawn at the narrow house width of 3.8 in; the three
curves differ by marker shape as well as hue.
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
BLUE, RED, AMBER, GREY = "#1f4e79", "#c0392b", "#d68910", "#8a9aa5"

d = json.load(open(_HERE / "ftns_figdata.json"))["f6"]

fig, a = plt.subplots(figsize=(3.8, 2.7))
a.loglog(d["d020"]["n"], d["d020"]["K"], "-s", color=RED, lw=1.4, ms=3.5, label=r"$d=0.20$")
a.loglog(d["d002"]["n"], d["d002"]["K"], "-^", color=AMBER, lw=1.4, ms=3.8, label=r"$d=0.02$")
a.loglog(d["d0"]["n"], d["d0"]["K"], "-o", color=BLUE, lw=1.4, ms=3.5, label=r"$d=0$ (null true)")
a.loglog(d["sqrtn"]["n"], d["sqrtn"]["K"], "--", color=GREY, lw=1.1, label=r"$\propto\sqrt{n}$")
a.axhline(d["hline"], color="k", lw=0.8, ls=":")
a.set_xlim(*d["xlim"]); a.set_ylim(*d["ylim"])
a.set_yticks([10.0 ** k for k in (2, -2, -6, -10, -14, -18, -22, -26, -30)])
a.set_xlabel(r"$n$ per genotype"); a.set_ylabel(r"Jeffreys' $K$ for $\kappa=1$")
a.set_title("refutable, never confirmable", fontsize=9)
a.legend(frameon=False, loc="lower left", labelspacing=0.3)
fig.tight_layout()
fig.savefig(str(_FIGDIR / "ftns_f6_bf.pdf"))
print("wrote", _FIGDIR / "ftns_f6_bf.pdf")
