"""Figure mp_theta: exact posterior for the mutation parameter theta under the Pochhammer
prior p(theta) propto 1/((theta+1)(theta+2)), likelihood theta^K Gamma(theta)/Gamma(theta+n),
against the point estimate solving E K_n(theta) = K_n.

Study: 400 replicates, K_n drawn by the Feller coupling, posterior on a 4000 point
geometric grid on [1e-2, 400].  Reports median 95% interval, its width relative to theta and
coverage at (n, theta) = (100, 5), (1000, 5), (100, 1).
Run: python -I fig_theta.py
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
from scipy.special import gammaln, digamma
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = str(_FIGDIR) + "/mp_theta.pdf"
GRID = np.geomspace(1e-2, 400, 4000)
R = 400


def style():
    for f in ["texgyrepagella-regular.otf", "texgyrepagella-italic.otf"]:
        p = "/usr/share/texmf/fonts/opentype/public/tex-gyre/" + f
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["TeX Gyre Pagella", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "font.size": 9,
                         "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "lines.linewidth": 1.1})


def feller_K(theta, n, size, rng):
    j = np.arange(n)
    return (rng.random((size, n)) < theta/(theta + j)).sum(1)


def EK(theta, n):
    return theta*(digamma(theta + n) - digamma(theta))


def posterior(K, n):
    """Normalised posterior density on GRID (trapezoid in theta)."""
    lp = K*np.log(GRID) + gammaln(GRID) - gammaln(GRID + n) - np.log((GRID + 1)*(GRID + 2))
    w = np.exp(lp - lp.max())
    w /= np.trapezoid(w, GRID)
    return w


def interval(dens, lev=0.95):
    cdf = np.concatenate([[0], np.cumsum(0.5*(dens[1:] + dens[:-1])*np.diff(GRID))])
    cdf /= cdf[-1]
    lo = np.interp((1 - lev)/2, cdf, GRID)
    hi = np.interp(1 - (1 - lev)/2, cdf, GRID)
    return lo, hi


def theta_hat(K, n):
    if K <= 1:
        return 0.0
    return brentq(lambda t: EK(t, n) - K, 1e-8, 1e6)


def study(theta, n, rng):
    Ks = feller_K(theta, n, R, rng)
    los, his = np.empty(R), np.empty(R)
    for i, K in enumerate(Ks):
        los[i], his[i] = interval(posterior(K, n))
    cover = np.mean((los <= theta) & (theta <= his))
    width = np.median(his - los)
    return dict(K=Ks, lo=np.median(los), hi=np.median(his), width=width,
                rel=width/theta, cover=cover)


def main():
    style()
    rng = np.random.default_rng(7)
    lines = []
    res = {}
    for theta, n in [(5, 100), (5, 1000), (1, 100)]:
        r = study(theta, n, rng)
        res[(theta, n)] = r
        lines.append(f"theta={theta} n={n}: median interval [{r['lo']:.2f}, {r['hi']:.2f}]  "
                     f"median width {r['width']:.2f} = {100*r['rel']:.0f}% of theta  "
                     f"coverage {r['cover']:.3f}  mean K {r['K'].mean():.2f}")
    # exact coverage from the exact law of K_n (Feller coupling convolution)
    for theta, n in [(5, 100), (5, 1000), (1, 100)]:
        pk = np.array([1.0])
        for j in range(n):
            q = theta/(theta + j)
            pk = np.convolve(pk, [1 - q, q])
        cov = sum(pk[k]*(lambda iv: iv[0] <= theta <= iv[1])(interval(posterior(k, n)))
                  for k in range(len(pk)) if pk[k] > 1e-12)
        lines.append(f"theta={theta} n={n}: exact coverage {cov:.3f}, "
                     f"P(K_n<=15)={pk[:16].sum():.3f}")
    # representative posterior: K = 15, the value whose interval is the median interval
    # quoted in the text (the law of K_100 at theta = 5 has its median on the 15/16 boundary,
    # P(K <= 15) = 0.485, so either value is a median replicate)
    n, theta = 100, 5
    K = 15
    dens = posterior(K, n)
    lo, hi = interval(dens)
    th = theta_hat(K, n)
    pmean = np.trapezoid(GRID*dens, GRID)
    lines.append(f"representative: K={K}, theta_hat={th:.2f}, posterior mean {pmean:.2f}, "
                 f"95% interval [{lo:.2f}, {hi:.2f}]")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "fig_theta_numbers.txt"), "w") as f:
        f.write(txt + "\n")

    fig, ax = plt.subplots(figsize=(3.8, 2.4))
    m = GRID <= 16
    ax.plot(GRID[m], dens[m], color="#0072B2", label="posterior")
    band = (GRID >= lo) & (GRID <= hi)
    ax.fill_between(GRID[band], 0, dens[band], color="#0072B2", alpha=0.18, lw=0,
                    label=r"95% interval")
    ax.axvline(th, color="#D55E00", ls="--", lw=1.0, label=r"point estimate $\hat\theta$")
    ax.axvline(theta, color="0.35", ls=":", lw=1.0, label=r"true $\theta=5$")
    ax.set_xlim(0, 16); ax.set_ylim(0, None)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("density")
    ax.text(0.98, 0.42, rf"$n={n}$, $K_n={K}$", transform=ax.transAxes, ha="right", fontsize=8)
    ax.legend(frameon=False, loc="upper right", handlelength=1.8, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
