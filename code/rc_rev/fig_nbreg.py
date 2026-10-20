"""Figure rc_nbreg (Figure rc:fig:nbreg, Section rc:sec:app), redrawn in house style from script outputs.

The original PDF had no script, and its panel (b) title printed alpha hat = 74.3, which is not a
maximum likelihood value (the Insurance profile likelihood increases to its Poisson limit; see
nbreg_insurance.txt, section 1). The old PDF is kept in misc/review_r2/baseline_figures/rc_nbreg.pdf.

Inputs (run these first):
  code/rc_rev/nbreg_alg2_draws.npz      from nbreg_alg2.py: quine draws of alpha under the three
                                        priors (Algorithm 2, 1e5 retained draws each); Insurance
                                        traces of Algorithm 2 and of the mean parameterization
                                        sampler (run 0, start alpha = 20)
  code/rc_rev/nbreg_insurance_draws.npz from nbreg_insurance.py: Insurance draws of alpha under the
                                        three priors for the (alpha, psi) model of Table rc:tab:nbreg
                                        (model A), 90 per cent predictive band ends per cell, and the
                                        posterior mean of each cell mean
Panels: (a) quine posteriors of alpha, with the maximum likelihood value 1.27 (MASS::glm.nb);
(b) Insurance posteriors of log10 alpha (no finite maximum likelihood value, so none is drawn);
(c) Insurance 90 per cent predictive bands under Ga(2,1) and PH(0,1,2,1), cells ordered by the
posterior mean of the cell mean under PH(0,1,2,1); (d) first 1500 retained draws of alpha on
Insurance for Algorithm 2 and the mean parameterization sampler, with their effective sample sizes
over all 20000 retained draws.

Run: python3 -I code/rc_rev/fig_nbreg.py   (writes figures/rc_nbreg.pdf, or $SMG_FIG_DIR/rc_nbreg.pdf)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "book.tex")):
    _ROOT = os.path.dirname(_ROOT)
_FIGDIR = os.environ.get("SMG_FIG_DIR", os.path.join(_ROOT, "figures"))
os.makedirs(_FIGDIR, exist_ok=True)
from nbreg_alg2 import ess  # noqa: E402

G = np.load(os.path.join(_HERE, "nbreg_alg2_draws.npz"))
I = np.load(os.path.join(_HERE, "nbreg_insurance_draws.npz"))

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
STY = {"PH": dict(ls="-", color="#2c5d8a", lw=1.2, label=r"PH$(0,1,2,1)$"),
       "Gdiff": dict(ls=":", color="0.35", lw=1.2, label=r"Ga$(0.01,0.01)$"),
       "G21": dict(ls="--", color="#a8322d", lw=1.2, label=r"Ga$(2,1)$")}
fig, ax = plt.subplots(2, 2, figsize=(5.4, 4.5))

a = ax[0, 0]
xs = np.linspace(0.7, 2.0, 400)
for k in ("PH", "Gdiff", "G21"):
    a.plot(xs, gaussian_kde(G["quine_" + k])(xs), **STY[k])
a.set_ylim(0, 3.9)
a.axvline(1.27, color="0.2", ls="-.", lw=0.8)
a.text(1.29, 0.12, r"$\hat\alpha$", fontsize=9)
a.set_xlabel(r"dispersion $\alpha$"); a.set_ylabel("posterior density")
a.set_title(r"(a) quine: $\hat\alpha=1.27$, priors agree", fontsize=9)
a.legend(frameon=False, fontsize=7, loc="upper right")

a = ax[0, 1]
xs = np.linspace(0.3, 5.0, 500)
for k in ("PH", "Gdiff", "G21"):
    a.plot(xs, gaussian_kde(np.log10(I["alpha_" + k].astype(float)))(xs), **STY[k])
a.set_xticks([1, 2, 3, 4, 5]); a.set_xticklabels([r"$10$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
a.set_xlabel(r"dispersion $\alpha$ (log scale)"); a.set_ylabel(r"density of $\log_{10}\alpha$")
a.set_title(r"(b) Insurance: $\hat\alpha=\infty$, priors disagree", fontsize=9)
a.legend(frameon=False, fontsize=7, loc="upper right")

a = ax[1, 0]
y = I["y"]; order = np.argsort(I["mubar_PH"]); xc = np.arange(len(y))
a.fill_between(xc, np.maximum(I["q05_G21"][order], 0.5), I["q95_G21"][order], color="#a8322d", alpha=0.18, lw=0)
a.plot(xc, I["q95_G21"][order], "--", color="#a8322d", lw=0.8, label=r"Ga$(2,1)$, 90% band")
a.plot(xc, np.maximum(I["q05_G21"][order], 0.5), "--", color="#a8322d", lw=0.8)
a.fill_between(xc, np.maximum(I["q05_PH"][order], 0.5), I["q95_PH"][order], color="#2c5d8a", alpha=0.35, lw=0)
a.plot(xc, I["q95_PH"][order], "-", color="#2c5d8a", lw=0.8, label=r"PH$(0,1,2,1)$, 90% band")
a.plot(xc, np.maximum(I["q05_PH"][order], 0.5), "-", color="#2c5d8a", lw=0.8)
a.plot(xc, np.maximum(y[order], 0.5), "o", ms=2.2, color="k", label="observed claims")
a.set_yscale("log"); a.set_ylim(0.5, 2e4)
a.set_xlabel("policy cell, ordered by fitted mean"); a.set_ylabel("claims")
a.set_title("(c) Insurance: 90% predictive bands", fontsize=9)
a.legend(frameon=False, fontsize=7, loc="upper left")

a = ax[1, 1]
for key, st, lab in (("ins_meanpar", dict(color="#7fa7cc", lw=0.5, ls="-"), "mean parameterization"),
                     ("ins_alg2", dict(color="#a8322d", lw=1.1, ls="-"), "Algorithm 2")):
    A = G[key]
    a.plot(np.arange(1500), A[:1500], label=f"{lab}, ESS = {ess(A):.0f}", **st)
a.set_yscale("log")
a.set_xlabel("iteration"); a.set_ylabel(r"$\alpha$")
a.set_title(r"(d) Insurance: mixing of $\alpha$", fontsize=9)
a.legend(frameon=False, fontsize=7, loc="upper left")
ymax = max(G["ins_meanpar"][:1500].max(), G["ins_alg2"][:1500].max())
a.set_ylim(None, ymax * 30)

fig.tight_layout()
out = os.path.join(_FIGDIR, "rc_nbreg.pdf")
fig.savefig(out)
print("wrote", out)
for key in ("ins_alg2", "ins_meanpar"):
    print(key, "ESS over 20000 retained draws:", round(ess(G[key])))
