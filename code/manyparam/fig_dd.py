"""Figure mp_dd: double descent and saturation of the optimism (Corollary saturate).

N = 200 individuals, ten causal variants of unit effect, sixty replicates.  Genotypes are
Bin(2, p_l) with p_l uniform on [0.1, 0.9], standardised by the true frequencies, so the
design is isotropic.  Residual variance sigma^2 = 1.  For each replicate one 200 x 800 genotype
matrix is drawn and its first p columns used, the ten causal variants being the first ten.

Left: out of sample (excess) prediction risk E(x'(betahat - beta))^2 = ||betahat - beta||^2
for the minimum norm least squares fit and for ridge with the penalty chosen by generalised
cross validation.  Right: plug in sigma_A^2 hat = ||H y||^2 / N, H the orthogonal projection
onto the column space of X, against sigma_A^2 + sigma^2 min(p, N)/N with sigma_A^2 = ||X beta||^2/N.
Run: python -I fig_dd.py
"""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = str(_FIGDIR) + "/mp_dd.pdf"
N, R, Q, SIG = 200, 60, 10, 1.0
RATIOS = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.6, 2.0, 3.0, 4.0])
PS = np.round(RATIOS*N).astype(int)
LAMS = np.logspace(-4, 4, 161)


def style():
    for f in ["texgyrepagella-regular.otf", "texgyrepagella-italic.otf"]:
        p = "/usr/share/texmf/fonts/opentype/public/tex-gyre/" + f
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["TeX Gyre Pagella", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "font.size": 9,
                         "axes.titlesize": 9, "axes.labelsize": 9,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "lines.linewidth": 1.1})


def fits(X, y, beta):
    U, d, Vt = np.linalg.svd(X, full_matrices=False)
    tol = d.max()*max(X.shape)*np.finfo(float).eps
    d = d[d > tol]; U = U[:, :len(d)]; Vt = Vt[:len(d)]
    uy = U.T @ y
    b_mn = Vt.T @ (uy/d)
    risk_mn = float(np.sum((b_mn - beta)**2))
    # ridge, penalty by GCV over a grid (lambda in units of N)
    lam = LAMS*N
    f = d[None, :]**2/(d[None, :]**2 + lam[:, None])            # shrink factors
    resid_in = np.sum(((1 - f)*uy[None, :])**2, 1) + (y @ y - uy @ uy)
    df = f.sum(1)
    gcv = N*resid_in/(N - df)**2
    k = int(np.argmin(gcv))
    b_r = Vt.T @ (d/(d**2 + lam[k])*uy)
    risk_r = float(np.sum((b_r - beta)**2))
    s2A_hat = float(uy @ uy)/N                                  # ||H y||^2 / N
    return risk_mn, risk_r, s2A_hat


def main():
    style()
    rng = np.random.default_rng(311)
    P = PS.max()
    risk_mn = np.zeros((R, len(PS))); risk_r = np.zeros_like(risk_mn)
    s2hat = np.zeros_like(risk_mn); s2true = np.zeros(R)
    for r in range(R):
        fr = rng.uniform(0.1, 0.9, P)
        G = rng.binomial(2, fr, size=(N, P)).astype(float)
        X = (G - 2*fr)/np.sqrt(2*fr*(1 - fr))
        b_full = np.zeros(P); b_full[:Q] = 1.0
        signal = X[:, :Q] @ b_full[:Q]
        y = signal + SIG*rng.standard_normal(N)
        s2true[r] = float(signal @ signal)/N
        for j, p in enumerate(PS):
            risk_mn[r, j], risk_r[r, j], s2hat[r, j] = fits(X[:, :p], y, b_full[:p])
    mn, rr = risk_mn.mean(0), risk_r.mean(0)
    sh = s2hat.mean(0); se = s2hat.std(0, ddof=1)/np.sqrt(R)
    pred = s2true.mean() + SIG**2*np.minimum(PS, N)/N
    below, above = RATIOS < 1, RATIOS >= 1
    lines = [f"p/N: {RATIOS.tolist()}",
             f"min norm risk: {np.round(mn, 2).tolist()}",
             f"ridge GCV risk: {np.round(rr, 2).tolist()}",
             f"sigma_A^2 hat mean: {np.round(sh, 2).tolist()}",
             f"per point MC s.e.: {np.round(se, 2).tolist()} (mean {se.mean():.2f})",
             f"mean true sigma_A^2 {s2true.mean():.2f}, s.d. across replicates {s2true.std(ddof=1):.2f}",
             f"group mean below threshold {sh[below].mean():.2f} vs predicted {pred[below].mean():.2f}",
             f"group mean at/above threshold {sh[above].mean():.2f} vs predicted {pred[above].mean():.2f}",
             f"isotropic theory risk at p/N=1.6: {10*(1-1/1.6) + SIG**2/(1.6-1):.2f}"]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "fig_dd_numbers.txt"), "w") as fh:
        fh.write(txt + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.3))
    ax[0].plot(RATIOS, mn, "o-", ms=2.5, color="#0072B2", label="minimum norm")
    ax[0].plot(RATIOS, rr, "s-", ms=2.5, color="#D55E00", label="ridge, GCV")
    ax[0].axvline(1, color="0.6", lw=0.7, ls=":")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$p/N$"); ax[0].set_ylabel("prediction risk")
    ax[0].legend(frameon=False, loc="upper right", borderaxespad=0.1)
    ax[1].plot(RATIOS, sh, "o", ms=2.5, color="#0072B2", label=r"$\widehat{\sigma}^2_A$")
    ax[1].plot(RATIOS, pred, "-", color="0.4", lw=0.9,
               label=r"$\sigma^2_A+\sigma^2\min(p,N)/N$")
    ax[1].axvline(1, color="0.6", lw=0.7, ls=":")
    ax[1].set_xlabel(r"$p/N$"); ax[1].set_ylabel(r"estimated $\sigma^2_A$")
    lo = min(sh.min(), pred.min()); hi = max(sh.max(), pred.max())
    ax[1].set_ylim(lo - 1.5, hi + 1.5)
    ax[1].legend(frameon=False, loc="lower right", borderaxespad=0.1)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
