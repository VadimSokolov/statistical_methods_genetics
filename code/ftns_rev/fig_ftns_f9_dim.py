"""Figure ftns_f9_dim: what the additive variance can see (Chapter ftns, Figure ftns:fig:dim).

Left: the percentage of the persistent state visible to the additive genetic variance, which
is L of the 2^L - 1 dimensions, 100 L / (2^L - 1) for L = 1 to 20. Right: the standard error
of a dominance contrast relative to its value at minor allele frequency 0.5, which is
0.25 / (p q); it is about 25 times larger at minor allele frequency 0.01. Both panels are
exact formulas, which reproduce the original figure. Redrawn at the house width of 5.4 in.
"""
import os as _os, sys as _sys
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

L = np.arange(1, 21)
vis = 100.0 * L / (2.0 ** L - 1)
p = np.linspace(0.01, 0.5, 400)
rel = 0.25 / (p * (1 - p))
print("visible at L=20: %.4f percent; relative se at MAF 0.01: %.2f" % (vis[-1], rel[0]))

fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.4))
a = ax[0]
a.semilogy(L, vis, "-o", color=BLUE, lw=1.4, ms=3.2)
a.annotate("one locus:\nnothing is hidden", xy=(1, 100), xytext=(4.5, 60), fontsize=8,
           va="center", arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
a.set_xticks([5, 10, 15, 20])
a.set_xlabel(r"number of loci $L$"); a.set_ylabel(r"percent of state visible to $\sigma^2_A$")
a.set_title(r"$L$ of $2^L-1$ dimensions", fontsize=9)

a = ax[1]
a.plot(p, rel, "-", color=RED, lw=1.8)
a.annotate(r"$25\times$ at MAF $0.01$", xy=(0.01, rel[0]), xytext=(0.12, 19), fontsize=8,
           va="center", arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
a.set_xlim(-0.015, 0.525); a.set_ylim(-0.6, 26.5); a.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
a.set_xlabel("minor allele frequency"); a.set_ylabel(r"se$(\hat D)$ relative to MAF $0.5$")
a.set_title(r"dominance contrasts: se $\propto 1/pq$", fontsize=9)
fig.tight_layout(w_pad=1.0)
fig.savefig(str(_FIGDIR / "ftns_f9_dim.pdf"))
print("wrote", _FIGDIR / "ftns_f9_dim.pdf")
