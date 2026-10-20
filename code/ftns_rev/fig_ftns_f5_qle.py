"""Figure ftns_f5_qle: quasi linkage equilibrium at s = 0.001 (Chapter ftns, Figure ftns:fig:qle).

Left: the disequilibrium D* against the recombination rate r for epsilon = 0.005, 0.01 and
0.02, with the approximation epsilon pA qA pB qB / r (dotted, eq. ftns:eq:Dstar). Right: the
relative error of Fisher's term divided by epsilon, which is flat at pA qA = 0.25 (dotted)
once the timescales separate. The plotted values were recovered from the original PDF by
extract_ftns_figdata.py and are read from ftns_figdata.json. Redrawn at the house width of
5.4 in; the three values of epsilon differ by marker shape as well as hue.
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
BLUE, RED, PURPLE = "#1f4e79", "#c0392b", "#6c3483"
SERIES = [("0.005", BLUE, "o", "-"), ("0.01", RED, "s", "--"), ("0.02", PURPLE, "^", "-.")]

d = json.load(open(_HERE / "ftns_figdata.json"))["f5"]

fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.4))
a = ax[0]
for key, col, mk, ls in SERIES:
    a.loglog(d[key]["r"], d[key]["Dstar"], ls, marker=mk, color=col, lw=1.3, ms=3.5,
             label=rf"$\epsilon={key}$")
a.loglog(d["theory"]["r"], d["theory"]["Dstar"], ":", color="k", lw=1.2,
         label=r"$\epsilon\,p_Aq_Ap_Bq_B/r$")
a.set_xlim(*d["xlim"]); a.set_ylim(2.5e-4, d["ylim_left"][1])  # lower floor leaves room for the legend
a.set_xlabel(r"recombination rate $r$"); a.set_ylabel(r"$D^\ast$")
a.set_title(r"disequilibrium falls like $1/r$", fontsize=9, color=BLUE)
a.legend(frameon=False, loc="lower left", handlelength=2.0, labelspacing=0.25, borderaxespad=0.2)

a = ax[1]
for key, col, mk, ls in SERIES:
    a.semilogx(d[key]["r_right"], d[key]["relerr"], ls, marker=mk, color=col, lw=1.3, ms=3.5,
               label=rf"$\epsilon={key}$")
a.axhline(d["flat"], color="k", lw=1.2, ls=":")
a.set_xlim(*d["xlim"]); a.set_ylim(*d["ylim_right"])
a.set_xlabel(r"recombination rate $r$"); a.set_ylabel(r"relative error $\div\ \epsilon$")
a.set_title("the relative error does not", fontsize=9, color=RED)
a.legend(frameon=False, loc="upper right", handlelength=2.2, labelspacing=0.3)
fig.tight_layout(w_pad=1.0)
fig.savefig(str(_FIGDIR / "ftns_f5_qle.pdf"))
print("wrote", _FIGDIR / "ftns_f5_qle.pdf")
