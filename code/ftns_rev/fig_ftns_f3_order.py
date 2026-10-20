"""Figure ftns_f3_order: the order of Fisher's remainder (Chapter ftns, Figure ftns:fig:order).

Left: the relative error |R|/(sigma_A^2/wbar) against the selection coefficient s, with the
reference line 0.28 s. Right: the sample size n* per genotype at which the neglected term
first exceeds the sampling noise, with the reference line 50 s^-4 and the size of the largest
assay ever run (10^4). The plotted values were recovered from the original PDF by
extract_ftns_figdata.py and are read from ftns_figdata.json. Redrawn at the house width of
5.4 in; each computed curve is solid and each reference line dotted.
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
BLUE, RED, GREY, PURPLE = "#1f4e79", "#c0392b", "#8a9aa5", "#6c3483"

d = json.load(open(_HERE / "ftns_figdata.json"))["f3"]
s = np.array(d["s"]); sr = np.array(d["s_right"])

fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.4))
a = ax[0]
a.loglog(s, d["rel_remainder"], "-", color=BLUE, lw=1.8)
a.loglog(s, d["line028"], ":", color=RED, lw=1.6, label=r"$0.28\,s$")
a.set_xlim(*d["xlim"]); a.set_ylim(*d["ylim_left"])
a.set_xlabel(r"selection coefficient $s$")
a.set_ylabel(r"$|\mathcal{R}|/(\sigma^2_A/\bar w)$")
a.set_title(r"the remainder is first order in $s$", fontsize=9)
a.legend(frameon=False, loc="upper left")

a = ax[1]
a.loglog(sr, d["nstar"], "-", color=PURPLE, lw=1.8)
a.loglog(sr, d["line50"], ":", color=RED, lw=1.6, label=r"$50\,s^{-4}$")
a.axhline(d["assay"], color=GREY, lw=1.0, ls="--")
a.text(sr[3], d["assay"] * 2.5, "largest assay ever run", fontsize=8, color="#5d6d7e")
a.set_xlim(*d["xlim"]); a.set_ylim(*d["ylim_right"])
a.set_yticks([1e4, 1e6, 1e8, 1e10, 1e12, 1e14])
a.set_xlabel(r"selection coefficient $s$")
a.set_ylabel(r"$n^\ast$ per genotype")
a.set_title("sample size needed to see it", fontsize=9)
a.legend(frameon=False, loc="upper right")
fig.tight_layout(w_pad=1.0)
fig.savefig(str(_FIGDIR / "ftns_f3_order.pdf"))
print("wrote", _FIGDIR / "ftns_f3_order.pdf")
