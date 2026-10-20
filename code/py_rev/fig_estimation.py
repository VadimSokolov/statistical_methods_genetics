"""Figure py_fig-estimation: estimating sigma = 0.5 from samples drawn with effort profile f_alpha.

Left, three estimators of the weighted exponent (log K_n / log n, 1 - 2 A_2 / A_1, A_1 / K_n)
against the truth sigma_* = 1/(alpha + 1). Right, the same three after the correction, with the
target sigma = 0.5 dashed. n = 10^6, 24 replicates. The simulated means were recovered from the
original PDF by extract_fig_data.py and are read from fig_data.json.
Writes figures/py_fig-estimation.pdf (or to SMG_FIG_DIR). Run: python3 -I fig_estimation.py
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
COL = ["#0072B2", "#D55E00", "#009E73"]
LS = ["-", "-.", (0, (1, 1.2))]
MK = ["o", "s", "^"]

D = json.loads((_HERE / "fig_data.json").read_text())["estimation"]
al = D["alpha"]
keys = ["logK", "spec2", "A1K"]
labs = [r"$\log K_n/\log n$", r"$1-2A_2/A_1$", r"$A_1/K_n$"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.55))
a1.plot(al, D["left"]["truth"], color="k", ls="--", lw=1.2, label=r"truth $\sigma_*$")
a2.axhline(0.5, color="k", ls="--", lw=1.2, label=r"truth $\sigma=0.5$")
for k, lab, c, ls, mk in zip(keys, labs, COL, LS, MK):
    a1.plot(al, D["left"][k], color=c, ls=ls, lw=1.3, marker=mk, ms=4, label=lab)
    a2.plot(al, D["right"][k], color=c, ls=ls, lw=1.3, marker=mk, ms=4, label="from " + lab)
for a in (a1, a2):
    a.set_xscale("log", base=2)
    a.set_xticks(al); a.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    a.minorticks_off()
    a.set_xlabel(r"effort index $\alpha$")
    a.grid(True, alpha=0.25, lw=0.5)
a1.set_ylabel(r"estimate of $\sigma_*$")
a1.set_title("(a) three estimators of\nthe weighted exponent", fontsize=9)
a1.legend(frameon=False, loc="lower left", handlelength=2.6, labelspacing=0.3, borderaxespad=0.2)
a2.set_ylim(0, 1)
a2.set_ylabel(r"corrected estimate of $\sigma$")
a2.set_title("(b) after the correction", fontsize=9)
a2.legend(frameon=False, loc="upper left", fontsize=8, handlelength=2.4, labelspacing=0.2, borderaxespad=0.1)
fig.tight_layout(w_pad=1.0)
_FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(str(_FIGDIR / "py_fig-estimation.pdf"))
plt.close(fig)
