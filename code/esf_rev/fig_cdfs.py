"""Figure esf_fig_cdfs: six realisations of the empirical distribution function F_500
under the Ewens predictive rule with H uniform on [0, 1], for theta = 1, 5, 25.

The original figure had no script in the repository. Its plotted staircases were
recovered exactly from the vector paths of the original PDF (now in
misc/review_r2/baseline_figures/) and cached in fig_cdfs_data.json beside this
script. This script redraws the same curves at house size (5.4 in wide). Each
realisation differs by line style as well as colour so that the figure reads in
black and white; the dashed black line is H.
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

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})

data = json.loads((_HERE / "fig_cdfs_data.json").read_text())["theta"]

# Muted colourblind safe colours (Paul Tol), each paired with its own line style.
styles = [("#332288", "-"), ("#CC6677", (0, (1, 1))), ("#117733", (0, (5, 1.5, 1, 1.5))),
          ("#AA4499", (0, (3, 1, 1, 1, 1, 1))), ("#44AA99", (0, (2, 1))),
          ("#999933", (0, (7, 1.5, 1, 1.5, 1, 1.5, 1, 1.5)))]

fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.0), sharey=True)
for ax, th in zip(axes, ("1", "5", "25")):
    for (col, ls), pts in zip(styles, data[th]):
        xy = np.array(pts)
        ax.plot(xy[:, 0], xy[:, 1], color=col, ls=ls, lw=0.9)
    ax.plot([0, 1], [0, 1], color="black", ls=(0, (3.7, 1.6)), lw=1.4, zorder=5)
    ax.set_title(r"$\theta = %s$" % th)
    ax.set_xlabel(r"$x$")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
axes[0].set_yticks([0, 0.5, 1])
axes[0].set_yticklabels(["0", "0.5", "1"])
axes[0].set_ylabel(r"$F_n(x)$")
fig.tight_layout(w_pad=0.8)
_FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(_FIGDIR / "esf_fig_cdfs.pdf")
print("wrote", _FIGDIR / "esf_fig_cdfs.pdf")
