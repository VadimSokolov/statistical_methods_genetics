"""Figure rc_ewens.pdf: (a) four genes, two alleles: half horseshoe prior and exact posteriors
under Ewens and Pitman Yor (sigma = 1/2); (b) Corbet's butterflies, posteriors for theta
under three priors. Round 2: Corbet's table is censored above 24 specimens (620 species in
all, 119 censored), so panel (b) conditions on n = n hat = 11068 and K = 620 as in Chapter esf
(code/esf_rev/butterflies_censored.py); the numbers are in butterflies_censored.txt."""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gammaln
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.3))
t = np.linspace(1e-4, 8, 2000)
log2, log3 = np.log(2), np.log(3)
prior = 1 / ((t + 1) * (t + 2) * log2)
C_e = log2 + 0.75 * log3 - 1.5
C_p = 0.5 * log2 + 5 / 8 * log3 - 1
post_e = t / ((t + 1) ** 2 * (t + 2) ** 2 * (t + 3)) / C_e
post_p = (t + 0.5) / ((t + 1) ** 2 * (t + 2) ** 2 * (t + 3)) / C_p
ax[0].plot(t, prior, color="0.55", ls="--", lw=1.0, label="prior")
ax[0].plot(t, post_e, color="C0", lw=1.3, label="Ewens")
ax[0].plot(t, post_p, color="C3", lw=1.3, label=r"Pitman Yor, $\sigma=1/2$")
ax[0].set_xlabel(r"$\theta$")
ax[0].set_ylabel("density")
ax[0].set_title(r"(a) $n=4$ genes, $K=2$ alleles", fontsize=9)
ax[0].legend(frameon=False, fontsize=7.5)
ax[0].set_xlim(0, 8); ax[0].set_ylim(0, 1.45)

n, K = 11068, 620          # n hat of the censored fit, all 620 species
g = np.linspace(80, 200, 4000)
ll = K * np.log(g) + gammaln(g) - gammaln(g + n)
priors = [("half horseshoe", -np.log(g + 1) - np.log(g + 2), "C0", "-"),
          (r"Ga$(0.1,0.1)$", -0.9 * np.log(g) - 0.1 * g, "C2", "-."),
          (r"Ga$(2,1)$", np.log(g) - g, "C3", "--")]
for name, lp, col, ls in priors:
    lg = ll + lp
    w = np.exp(lg - lg.max()); w /= np.trapezoid(w, g)
    ax[1].plot(g, w, color=col, ls=ls, lw=1.3, label=name)
ax[1].axvline(141.75, color="0.4", lw=0.8, ls=":")
ax[1].text(143.5, 0.085, "MLE", fontsize=7.5, color="0.3")
ax[1].set_xlabel(r"$\theta$")
ax[1].set_title(r"(b) butterflies, $\hat n=11068$, $K=620$", fontsize=9)
ax[1].legend(frameon=False, fontsize=7.5, loc="upper left")
ax[1].set_ylim(0, 0.13); ax[1].set_xlim(90, 175)
fig.tight_layout()
fig.savefig(str(_FIGDIR) + "/rc_ewens.pdf")
print("saved")
