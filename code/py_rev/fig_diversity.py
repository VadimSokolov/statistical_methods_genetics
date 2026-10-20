"""Figure py_fig-diversity: the diversity map at sigma = 0.5, theta = 1, alpha = 2 (kappa = 4/3).

Left, the weighted diversity S_* against the unweighted S on 220 common stick realisations, with
the predicted curve C S^kappa. Right, the two density histograms (28 bins each). The simulated
points, the curve and the histogram bars were recovered from the original PDF by
extract_fig_data.py and are read from fig_data.json; the bars are redrawn exactly as plotted.
Writes figures/py_fig-diversity.pdf (or to SMG_FIG_DIR). Run: python3 -I fig_diversity.py
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
                     "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 8.5,
                     "hatch.linewidth": 0.6})
BLUE, RED = "#0072B2", "#D55E00"

D = json.loads((_HERE / "fig_data.json").read_text())["diversity"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.45))
a1.scatter(D["S"], D["Sstar"], s=5, color=BLUE, alpha=0.6, lw=0, label="simulated")
a1.plot(D["curve_S"], D["curve_Sstar"], color=RED, lw=1.5, label=r"$CS^{\kappa}$, $\kappa=1.333$")
a1.set_xlabel(r"$S$ (unweighted)"); a1.set_ylabel(r"$S_*$ (weighted)")
a1.set_xticks(range(1, 7)); a1.set_yticks(range(0, 11, 2))
a1.grid(True, alpha=0.25, lw=0.5)
a1.set_title(r"(a) $\sigma=0.5,\ \theta=1,\ \alpha=2$", fontsize=9)
a1.legend(frameon=False, loc="upper left", markerscale=2, borderaxespad=0.2)


def bars(ax, rows, **kw):
    ax.bar([b[0] for b in rows], [max(b[2], 0.0) for b in rows], width=[b[1] - b[0] for b in rows],
           align="edge", **kw)


bars(a2, D["hist_S"], color=BLUE, alpha=0.5, lw=0, label=r"$S$")
bars(a2, D["hist_Sstar"], facecolor="none", edgecolor=RED, hatch="////", lw=0.8, label=r"$S_*$")
a2.set_xlabel("diversity"); a2.set_ylabel("density")
a2.set_xticks(range(0, 9, 2))
a2.grid(True, alpha=0.25, lw=0.5)
a2.set_title("(b) the weighting is a power\ntransform, not a new family", fontsize=9)
a2.legend(frameon=False, loc="upper right", borderaxespad=0.2)
fig.tight_layout(w_pad=1.0)
_FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(str(_FIGDIR / "py_fig-diversity.pdf"))
plt.close(fig)
