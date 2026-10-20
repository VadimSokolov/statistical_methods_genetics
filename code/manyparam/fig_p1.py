"""Figure mp_p1.

Left: estimated over true additive genetic variance against N, L = 400 loci, 40 replicates
at N in {200, 400, 800, 1600, 3200}.  Design (Section Simulation design): allele frequencies
uniform on [0.1, 0.9], genotypes Bin(2, p_l) standardised, twenty causal loci with standard
normal effects, an epistatic term formed from ten products of disjoint pairs of causal loci
scaled by 0.3, environmental noise standard normal.  The loci are unlinked, so N^{-1} X'X is
the identity up to O(N^{-1/2}) and the fitted projection is taken in its orthogonal design form
betahat = X'y/N, sigma_A^2 hat = ||betahat||^2 (true sigma_A^2 = ||beta||^2).
Horseshoe (species sampling prior, alpha = 1/2) posterior mean of each coefficient by a 400
point midpoint rule in kappa on (0,1), normal means model betahat_j ~ N(beta_j, s^2),
beta_j | lambda_j ~ N(0, s^2 tau^2 lambda_j^2), lambda_j half Cauchy, s estimated by the median
absolute deviation of the coefficients.  Global scale: canonical tau = L^{-1/2} (in units of s)
and oracle tau = sqrt(q/L), q = 20.

Right: rare variant discovery.  Feller coupling realisation of size 2n, K_n read off the first n,
theta hat solving E K_n(theta) = K_n, prediction E K_2n(theta hat); 300 replicates for
(theta, n) in {2, 10} x {200, 1000}.
Run: python -I fig_p1.py
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
from scipy.special import digamma
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = str(_FIGDIR) + "/mp_p1.pdf"
L, Q, NPAIR, EPI = 400, 20, 10, 0.3
NS = np.array([200, 400, 800, 1600, 3200])
R_A = 40
R_C = 300
KAP = (np.arange(400) + 0.5)/400          # midpoint rule in kappa


def style():
    for f in ["texgyrepagella-regular.otf", "texgyrepagella-italic.otf"]:
        p = "/usr/share/texmf/fonts/opentype/public/tex-gyre/" + f
        if not os.path.exists(p):   # fall back to the local TeX installation (macOS, TinyTeX)
            import subprocess
            try:
                p = subprocess.run(["kpsewhich", f], capture_output=True, text=True).stdout.strip()
            except OSError:
                p = ""
        if p and os.path.exists(p):
            font_manager.fontManager.addfont(p)
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["TeX Gyre Pagella", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "font.size": 9,
                         "axes.titlesize": 9, "axes.labelsize": 9,
                         "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "lines.linewidth": 1.1})


# ---------------------------------------------------------------- additive genetic variance
def hs_postmean(z, s, tau):
    """Horseshoe posterior mean (1 - E[kappa|z]) z by midpoint quadrature in kappa.
    Prior density of kappa = 1/(1 + tau^2 lambda^2), lambda half Cauchy:
    p(kappa) = (tau/pi) kappa^{-1/2} (1-kappa)^{-1/2} / (1 - (1 - tau^2) kappa).
    Likelihood of z given kappa: N(0, s^2/kappa)."""
    lp = (-0.5*np.log(KAP) - 0.5*np.log1p(-KAP) - np.log1p(-(1 - tau**2)*KAP))[None, :]
    ll = 0.5*np.log(KAP)[None, :] - 0.5*KAP[None, :]*(z[:, None]/s)**2
    w = lp + ll
    w = np.exp(w - w.max(1, keepdims=True))
    Ek = (w*KAP[None, :]).sum(1)/w.sum(1)
    return (1 - Ek)*z


def one_rep(N, rng):
    fr = rng.uniform(0.1, 0.9, L)
    G = rng.binomial(2, fr, size=(N, L)).astype(float)
    sd = G.std(0)
    sd[sd == 0] = 1.0
    X = (G - G.mean(0))/sd
    causal = rng.choice(L, Q, replace=False)
    beta = np.zeros(L); beta[causal] = rng.standard_normal(Q)
    pairs = causal.reshape(NPAIR, 2)
    epi = EPI*sum(X[:, i]*X[:, j] for i, j in pairs)
    y = X @ beta + epi + rng.standard_normal(N)
    y = y - y.mean()
    s2A = float(beta @ beta)
    bhat = X.T @ y/N
    s = np.median(np.abs(bhat - np.median(bhat)))/0.6745
    proj = float(bhat @ bhat)
    can = hs_postmean(bhat, s, L**-0.5)
    ora = hs_postmean(bhat, s, np.sqrt(Q/L))
    resid_var = float(np.var(y - X @ beta))            # sigma^2: epistasis plus environment
    return proj/s2A, float(can @ can)/s2A, float(ora @ ora)/s2A, resid_var, s2A, N*s**2


def study_A(rng):
    out = np.zeros((len(NS), R_A, 6))
    for i, N in enumerate(NS):
        for r in range(R_A):
            out[i, r] = one_rep(N, rng)
    return out


# ---------------------------------------------------------------- rare variant discovery
def EK(theta, n):
    return theta*(digamma(theta + n) - digamma(theta))


def theta_hat(K, n):
    if K <= 1:
        return 1e-8
    return brentq(lambda t: EK(t, n) - K, 1e-8, 1e7)


def study_C(rng):
    res = {}
    for theta in (2, 10):
        for n in (200, 1000):
            j = np.arange(2*n)
            B = rng.random((R_C, 2*n)) < theta/(theta + j)
            Kn = B[:, :n].sum(1); K2n = B.sum(1)
            pred = np.array([EK(theta_hat(k, n), 2*n) for k in Kn])
            res[(theta, n)] = dict(Kn=Kn, K2n=K2n, pred=pred,
                                   relerr=np.mean(np.abs(pred - K2n)/K2n),
                                   naive=np.mean(2*Kn)/np.mean(K2n))
    return res


def main():
    style()
    rng = np.random.default_rng(1930)
    A = study_A(rng)
    C = study_C(rng)
    m = A.mean(1)
    lines = ["N: " + str(NS.tolist()),
             "unregularised projection:      " + str(np.round(m[:, 0], 2).tolist()),
             "species sampling, tau=L^-1/2:  " + str(np.round(m[:, 1], 2).tolist()),
             "species sampling, oracle tau:  " + str(np.round(m[:, 2], 2).tolist()),
             "gap canonical vs oracle:       " + str(np.round(m[:, 2] - m[:, 1], 2).tolist()),
             "mean residual var (epi+env):   " + str(np.round(m[:, 3], 3).tolist()),
             "mean true sigma_A^2:           " + str(np.round(m[:, 4], 2).tolist()),
             "N s^2 (coef noise var x N):    " + str(np.round(m[:, 5], 2).tolist()),
             "MC s.e. of projection ratio:   " + str(np.round(A[:, :, 0].std(1, ddof=1)/np.sqrt(R_A), 3).tolist()),
             "prior effective model size L tau/(1+tau) at tau=L^-1/2: "
             f"{L*L**-0.5/(1 + L**-0.5):.2f}"]
    for key, r in C.items():
        lines.append(f"theta={key[0]} n={key[1]}: K_n {r['Kn'].mean():.2f}  predicted K_2n "
                     f"{r['pred'].mean():.2f}  actual K_2n {r['K2n'].mean():.2f}  "
                     f"mean rel err {100*r['relerr']:.1f}%  naive 2K_n / K_2n {r['naive']:.2f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "fig_p1_numbers.txt"), "w") as fh:
        fh.write(txt + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.4))
    a = ax[0]
    a.axhline(1, color="0.6", lw=0.7, ls=":")
    # Corollary reference: 1 + p sigma^2/(N sigma_A^2), sigma^2 the per observation noise of
    # the fitted coefficients (N s^2), averaged over replicates
    ref = 1 + L*(A[:, :, 5]/A[:, :, 4]).mean(1)/NS
    nn = np.geomspace(NS[0], NS[-1], 100)
    c_ref = np.polyfit(1/NS, ref - 1, 1)
    a.plot(nn, 1 + np.polyval(c_ref, 1/nn), color="0.5", lw=0.8, ls="--",
           label=r"$1+p\sigma^2/(N\sigma^2_A)$")
    a.plot(NS, m[:, 0], "o-", ms=2.5, color="#D55E00", label="unregularised projection,\northogonal design form")
    a.plot(NS, m[:, 1], "s-", ms=2.5, color="#0072B2", label=r"prior, $\tau=\sigma L^{-1/2}$")
    a.plot(NS, m[:, 2], "^-", ms=2.5, color="#009E73", label=r"prior, oracle $\tau$")
    a.set_xscale("log")
    a.set_xticks(NS); a.set_xticklabels([str(n) for n in NS])
    a.minorticks_off()
    a.set_xlabel(r"sample size $N$")
    a.set_ylabel(r"estimated over true $\sigma^2_A$")
    a.set_ylim(0, 4.2)   # headroom so the legend clears the curves
    a.legend(frameon=False, loc="upper right", borderaxespad=0.1, handlelength=1.6)

    b = ax[1]
    cols = {(2, 200): "#0072B2", (2, 1000): "#56B4E9", (10, 200): "#D55E00", (10, 1000): "#E69F00"}
    lo, hi = np.inf, 0
    for key, r in C.items():
        jit = rng.uniform(-0.3, 0.3, R_C)
        b.plot(r["K2n"] + jit, r["pred"], "o", ms=1.4, alpha=0.35, color=cols[key], mec="none")
        b.plot(r["K2n"].mean(), r["pred"].mean(), "o", ms=4, color=cols[key], mec="k", mew=0.4,
               label=rf"$\theta={key[0]}$, $n={key[1]}$")
        # naive doubling rule, mean of 2 K_n, hollow marker
        b.plot(r["K2n"].mean(), 2*r["Kn"].mean(), "o", ms=4, mfc="none", mec=cols[key], mew=0.8,
               label=r"naive $2K_n$" if key == (10, 1000) else None)
        lo = min(lo, r["K2n"].min(), r["pred"].min()); hi = max(hi, r["K2n"].max(), r["pred"].max())
    hi = max(hi, max(2*r["Kn"].mean() for r in C.values()))
    g = np.array([0, hi*1.05])
    b.plot(g, g, color="0.4", lw=0.8)
    b.set_xlim(0, hi*1.05); b.set_ylim(0, hi*1.05)
    b.set_xlabel(r"actual $K_{2n}$")
    b.set_ylabel(r"predicted $K_{2n}$")
    b.legend(frameon=False, loc="lower right", borderaxespad=0.1, handletextpad=0.2)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
