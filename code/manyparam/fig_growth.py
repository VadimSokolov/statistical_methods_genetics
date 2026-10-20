"""Figure mp_growth: expected number of occupied blocks K_n under the two parameter
Chinese restaurant process matched (Theorem main) to kappa ~ Be(1/2,1/2), Be(1,1), Be(1,3),
i.e. PD(1/2,0), PD(0,1), PD(0,3).  60 replicates, run to n = 3e4, six checkpoints.

K_n alone is a Markov chain: P(new block at customer n+1 | K_n = k) = (theta + alpha k)/(theta + n),
so the CRP block count is simulated exactly without tracking block sizes.
Run: python -I fig_growth.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = str(_FIGDIR) + "/mp_growth.pdf"


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
                         "axes.titleweight": "normal", "axes.labelweight": "normal",
                         "lines.linewidth": 1.1})


C = {"hs": "#0072B2", "sb": "#D55E00", "r3": "#009E73"}
CHECK = np.array([100, 300, 1000, 3000, 10000, 30000])
NMAX = CHECK[-1]
R = 60


def crp_counts(alpha, theta, rng):
    K = np.zeros(R)
    out = np.zeros((len(CHECK), R))
    ci = 0
    for n in range(NMAX):           # customer n+1 arrives, n already seated
        pnew = np.ones(R) if n == 0 else (theta + alpha*K)/(theta + n)
        K += rng.random(R) < pnew
        if n + 1 == CHECK[ci]:
            out[ci] = K
            ci += 1
    return out


def EK_exact(alpha, theta, n):
    n = np.asarray(n, float)
    if alpha == 0:
        return theta*(digamma(theta + n) - digamma(theta))
    # alpha > 0, theta = 0: E K_n = Gamma(n+alpha)/(alpha Gamma(alpha) Gamma(n))
    return np.exp(gammaln(n + alpha) - gammaln(alpha + 1) - gammaln(n))


def main():
    style()
    rng = np.random.default_rng(20251)
    cases = [("hs", 0.5, 0.0, r"$\mathrm{Be}(1/2,1/2)$, $\mathrm{PD}(1/2,0)$"),
             ("sb", 0.0, 1.0, r"$\mathrm{Be}(1,1)$, $\mathrm{PD}(0,1)$"),
             ("r3", 0.0, 3.0, r"$\mathrm{Be}(1,3)$, $\mathrm{PD}(0,3)$")]
    res = {}
    lines = []
    for key, a, th, lab in cases:
        K = crp_counts(a, th, rng)
        m = K.mean(1)
        res[key] = m
        sl_loglog = np.polyfit(np.log(CHECK), np.log(m), 1)[0]
        sl_log = np.polyfit(np.log(CHECK), m, 1)[0]
        lines.append(f"{key}: means {np.round(m, 2).tolist()}  loglog slope {sl_loglog:.3f}  "
                     f"slope in log n {sl_log:.3f}  exact E K at 3e4 {EK_exact(a, th, NMAX):.2f}  "
                     f"exact at 1e6 {EK_exact(a, th, 1e6):.1f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "fig_growth_numbers.txt"), "w") as f:
        f.write(txt + "\n")

    nn = np.logspace(2, np.log10(NMAX), 200)
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.3))
    for key, a, th, lab in cases:
        for j in range(2):
            ax[j].plot(nn, EK_exact(a, th, nn), color=C[key], lw=0.8, alpha=0.6)
            ax[j].plot(CHECK, res[key], "o", ms=3, color=C[key], label=lab if j == 0 else None)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"sample size $n$"); ax[0].set_ylabel(r"$\mathbb{E}K_n$")
    ax[0].set_title("log log axes", fontsize=9)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"sample size $n$ (log scale)")
    ax[1].set_title(r"against $\log n$", fontsize=9)
    ax[1].set_ylim(0, 60)
    ax[0].legend(frameon=False, loc="upper left", handletextpad=0.3, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
