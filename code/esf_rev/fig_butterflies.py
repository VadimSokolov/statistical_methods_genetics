"""The five butterfly figures of the esf chapter, redrawn after the censored refit (esf_a-06).

  esf_fig_butterflies.pdf   observed counts against the Ewens means at theta 141.75, n 11068
  esf_fig_weighted.pdf      power weighted paintbox spectra (theta 142, n 11068, 60 replicates)
  esf_fig_butterflies2.pdf  observed counts, Ewens and two parameter means
                            (alpha -0.1454, theta 209.70, n 9747)
  esf_fig_posterior.pdf     left: Ewens posterior for theta at n hat, quadrature against the
                            Beta augmentation sampler; right: joint (alpha, log theta)
                            posterior of the two parameter model with n integrated out
  esf_fig_predictive.pdf    posterior predictive number of new species in a second sample of
                            the same size, both models, Good and Toulmin's 75 marked

The old figures had no script; their PDFs are kept in misc/review_r2/baseline_figures/.
Values come from butterflies_refit.json (fits and means) and fig_butterflies_data.npz
(written by fig_butterflies_data.py: weighted spectra, posterior curves and draws).
Run: python3 -I code/esf_rev/fig_butterflies.py   (seconds).
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
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "legend.fontsize": 8,
                     "figure.constrained_layout.use": True})

# Paul Tol muted palette
INDIGO, ROSE, GREEN, TEAL, SAND, WINE = "#332288", "#CC6677", "#117733", "#44AA99", "#DDCC77", "#882255"
W_NARROW = 4.05      # 0.75 \textwidth of the 5.4 in text block
W_FULL = 5.4

obs = np.array([118, 74, 44, 24, 29, 22, 20, 19, 20, 15, 12, 14,
                6, 12, 6, 9, 9, 6, 10, 10, 11, 5, 3, 3], dtype=float)
jv = np.arange(1, 25)
RES = json.loads((_HERE / "butterflies_refit.json").read_text())
D = np.load(_HERE / "fig_butterflies_data.npz")
_FIGDIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(_FIGDIR / name)   # native size = figsize
    plt.close(fig)
    print("wrote", _FIGDIR / name)


def observed(ax):
    ax.plot(jv, obs, ls="none", marker="o", ms=3.5, color=ROSE, label="observed", zorder=5)


def spectrum_axes(ax, ylabel=r"$A_j$, number of species"):
    ax.set_xlabel(r"$j$, individuals per species")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 25)


# ---------------------------------------------------------------- Figure butterflies
th0 = RES["ewens"]["theta"]; n0 = RES["ewens"]["n"]
fig, ax = plt.subplots(figsize=(W_NARROW, 2.5))
observed(ax)
ax.plot(jv, RES["table"]["ewens"], color=INDIGO, lw=1.2,
        label=r"Ewens means, $\hat\theta = %.2f$, $\hat n = %d$" % (th0, n0))
spectrum_axes(ax); ax.set_ylim(0, 150)
ax.legend(loc="upper right")
save(fig, "esf_fig_butterflies.pdf")

# ---------------------------------------------------------------- Figure weighted
styles = [(INDIGO, "-", None), (TEAL, (0, (5, 2)), None), (GREEN, (0, (5, 1.5, 1, 1.5)), None), (WINE, (0, (1, 1.2)), None)]
fig, ax = plt.subplots(figsize=(W_NARROW, 2.6))
for (col, ls, _), beta, sp in zip(styles, D["betas"], D["spec"]):
    ax.plot(jv, sp[:24], color=col, ls=ls, lw=1.1, label=r"$\beta = %.2f$" % beta)
observed(ax)
ax.set_yscale("log")
spectrum_axes(ax, r"$A_j$ (log scale)")
ax.legend(loc="upper right", ncol=1)
save(fig, "esf_fig_weighted.pdf")

# ---------------------------------------------------------------- Figure butterflies2
tp = RES["twoparam"]
fig, ax = plt.subplots(figsize=(W_NARROW, 2.5))
observed(ax)
ax.plot(jv, RES["table"]["ewens"], color=INDIGO, lw=1.2, label=r"Ewens, $\hat\theta = %.2f$" % th0)
ax.plot(jv, tp["means"], color=TEAL, lw=1.4, ls=(0, (5, 2)),
        label=r"two parameter, $\hat\alpha = %.3f$, $\hat m = %d$" % (tp["alpha"], round(tp["m"])))
spectrum_axes(ax); ax.set_ylim(0, 150)
ax.legend(loc="upper right")
save(fig, "esf_fig_butterflies2.pdf")

# ---------------------------------------------------------------- Figure posterior
fig, (axl, axr) = plt.subplots(1, 2, figsize=(W_FULL, 2.2))
axl.hist(D["draws"], bins=np.arange(110, 176, 1.5), density=True, color="#88CCEE", alpha=0.75,
         edgecolor="none", label="Beta augmentation sampler")
axl.plot(D["tq"], D["dens_q"], color=INDIGO, lw=1.2, label="quadrature")
axl.set_xlim(112, 172); axl.set_ylim(0, 0.085)
axl.set_xlabel(r"$\theta$"); axl.set_ylabel("posterior density")
axl.legend(loc="upper left", fontsize=7, handlelength=1.5)
ag, Pat, lt = D["agrid"], D["Pat"], np.log(D["thg"])
dens = Pat / np.gradient(ag)[:, None]                 # per unit alpha; log theta grid is uniform
ae = np.concatenate([[ag[0] - (ag[1] - ag[0]) / 2], (ag[1:] + ag[:-1]) / 2, [ag[-1] + (ag[-1] - ag[-2]) / 2]])
te = np.concatenate([[lt[0] - (lt[1] - lt[0]) / 2], (lt[1:] + lt[:-1]) / 2, [lt[-1] + (lt[-1] - lt[-2]) / 2]])
axr.pcolormesh(ae, te, dens.T, cmap="Greys", shading="flat", rasterized=True)
axr.set_xlim(-0.38, 0.06); axr.set_ylim(4.85, 5.75)
axr.set_xlabel(r"$\alpha$"); axr.set_ylabel(r"$\log\theta$")
save(fig, "esf_fig_posterior.pdf")

# ---------------------------------------------------------------- Figure predictive
pe, pp = D["pe"], D["pp"]
bins = np.arange(20.5, 161.5, 3)
fig, ax = plt.subplots(figsize=(W_NARROW, 2.5))
ax.hist(pp, bins=bins, density=True, color=TEAL, alpha=0.35, edgecolor="none")
ax.hist(pp, bins=bins, density=True, histtype="step", color=TEAL, lw=1.1, ls=(0, (4, 1.5)),
        label="two parameter, mean %.1f" % pp.mean())
ax.hist(pe, bins=bins, density=True, color=SAND, alpha=0.45, edgecolor="none")
ax.hist(pe, bins=bins, density=True, histtype="step", color=INDIGO, lw=1.1,
        label="Ewens, mean %.1f" % pe.mean())
ax.axvline(75, color="black", ls=(0, (1, 1.5)), lw=1.2, label="Good and Toulmin, 75")
ax.set_xlim(25, 180); ax.set_ylim(0, 0.047)
ax.set_xlabel("new species in a second sample of the same size")
ax.set_ylabel("posterior predictive density")
ax.legend(loc="upper right", handlelength=1.8)
save(fig, "esf_fig_predictive.pdf")
