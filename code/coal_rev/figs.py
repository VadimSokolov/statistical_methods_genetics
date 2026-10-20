"""Figures for the coal chapter revision: exact law of S at n=4 against coalescent simulation; SBC histograms."""
import sys, numpy as np
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from coalsim import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

rng = np.random.default_rng(44)
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.2))
for ax, th in zip(axes, [1.0, 4.0]):
    R = 100000
    S = np.array([simulate(4, th, rng)["S"] for _ in range(R)])
    smax = 10 if th == 1 else 25
    ex = pmf_S(4, th, smax)
    freq = np.array([np.mean(S == s) for s in range(smax+1)])
    ax.bar(np.arange(smax+1), ex, color="0.8", width=0.8, label="exact law")
    ax.plot(np.arange(smax+1), freq, "o", ms=3, color="k", label="simulated coalescent")
    ax.set_xlabel("number of segregating sites $s$")
    ax.set_title(r"$n=4,\ \theta=%g$" % th)
    print(th, "max abs diff", np.abs(freq-ex).max())
axes[0].set_ylabel(r"$P(S=s)$")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(str(_FIGDIR) + "/coal_pmf4.pdf")

d = np.load(str(_HERE / "sbc.npz"))
fig, axes = plt.subplots(1, 4, figsize=(5.4, 1.8), sharey=True)
titles = ["exact given $S$", "prior as posterior", "Poisson gamma", "recalibrated"]
for ax, key, t in zip(axes, ["u_exact", "u_prior", "u_pg", "u_rc"], titles):
    c = np.histogram(d[key], 10, (0, 1))[0]
    ax.bar(np.arange(10)/10+0.05, c/len(d[key])*10, width=0.09, color="0.6")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_title(t, fontsize=8)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("rank / PIT")
axes[0].set_ylabel("density")
fig.tight_layout()
fig.savefig(str(_FIGDIR) + "/coal_sbc.pdf")
