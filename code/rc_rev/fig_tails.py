"""Figure rc_tails (Figure rc:fig:tails, Sections rc:sec:exact and rc:sec:nb), redrawn in house style.

The original PDF had no script; this script reproduces both panels from their description in the
chapter (the old PDF is kept in misc/review_r2/baseline_figures/rc_tails.pdf).

Left: log10 kappa_n for the Dirichlet process posterior at rho = k/n = 1/8, that is the Ewens
likelihood theta^k/[theta]^n times the half horseshoe prior PH(0,1,2,1) proportional to
1/((theta+1)(theta+2)), with k = ceil(n/8), n = 10, 20, ..., 160. kappa_n = max_j |gamma_j| / C,
the largest partial fraction coefficient of the kernel over its integral C (as in illcond.py).
The poles at 1 and 2 are double; their coefficients come from derivatives at 150 digits, and C
from quadrature. This convention reproduces the end points of the old panel (about 4 at n = 10
and 78 at n = 160); the old quoted fit 0.498 n - 2.67 could not be reproduced by any of the
conventions tried (k = n//8, round(n/8), ceil(n/8); with no prior, PH(0,1,2,1) or PH(0,1/2,2,1);
max or sum of the coefficients), which give slopes 0.502 to 0.520. The fit printed here is the
one the chapter now quotes.

Right: marginal p(n) of the hierarchy alpha ~ PH(0,a,2,1), pi | alpha ~ Be(alpha + a, 2),
n | alpha, pi ~ NB(alpha, pi), so p(n | alpha) = [alpha]^n/n! B(2 alpha + a, n + 2)/B(alpha + a, 2),
integrated over alpha by quadrature in logarithms, for a = 1, 2, against the asymptote
K n^{-(1+a)} (log n)^{-2} of Theorem rc:thm:zipf, K = Gamma(a) p_alpha(0)/B(a, 2),
p_alpha(0) = 1/(a (a+1) log((a+1)/a)). Prints the ratio p(n) n^{1+a} (log n)^2 / K.

Run: python3 -I code/rc_rev/fig_tails.py   (writes figures/rc_tails.pdf, or $SMG_FIG_DIR/rc_tails.pdf)
"""
import os
import sys
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while not os.path.exists(os.path.join(_ROOT, "book.tex")):
    _ROOT = os.path.dirname(_ROOT)
_FIGDIR = os.environ.get("SMG_FIG_DIR", os.path.join(_ROOT, "figures"))
os.makedirs(_FIGDIR, exist_ok=True)

mp.mp.dps = 150
PRIOR = (1, 2)


def log10kappa(n, k):
    poles = {}
    for r in list(range(1, n)) + list(PRIOR):
        poles[r] = poles.get(r, 0) + 1
    cmax = mp.mpf(0)
    for r, mult in poles.items():
        def g(t, r=r):
            v = t ** (k - 1)
            for q, mq in poles.items():
                if q != r:
                    v /= (t + q) ** mq
            return v
        for j in range(mult):
            c = g(mp.mpf(-r)) if j == 0 else mp.diff(g, mp.mpf(-r), j) / mp.factorial(j)
            cmax = max(cmax, abs(c))
    f = lambda t: t ** (k - 1) / mp.fprod([t + i for i in range(1, n)]) / mp.fprod([t + r for r in PRIOR])
    C = mp.quad(f, [0, 0.5, 2, 5, 10, 20, 50, 200, mp.inf])
    return float(mp.log10(cmax / C))


ns = np.arange(10, 161, 10)
lk = np.array([log10kappa(int(n), int(np.ceil(n / 8))) for n in ns])
s, b = np.polyfit(ns, lk, 1)
r2 = 1 - np.sum((lk - (s * ns + b)) ** 2) / np.sum((lk - lk.mean()) ** 2)
print("n:", ns.tolist())
print("log10 kappa_n:", np.round(lk, 2).tolist())
print(f"least squares fit log10 kappa_n = {s:.3f} n {b:+.2f}, R^2 = {r2:.5f}; double precision (16 digits) exhausted at n = {(16 - b) / s:.1f}")

mp.mp.dps = 30


def p_marg(n, a):
    Z = mp.log(mp.mpf(a + 1) / a)
    lB0 = lambda al: mp.loggamma(al + a) + mp.loggamma(2) - mp.loggamma(al + a + 2)
    def h(al):
        lp = (mp.loggamma(al + n) - mp.loggamma(al) - mp.loggamma(n + 1)
              + mp.loggamma(2 * al + a) + mp.loggamma(n + 2) - mp.loggamma(2 * al + a + n + 2) - lB0(al))
        return mp.exp(lp) / ((al + a) * (al + a + 1) * Z)
    L = mp.log(n)
    pts = [0] + [mp.mpf(x) / L for x in (0.25, 0.5, 1, 2, 4, 8, 16, 40)] + [1, 10, mp.inf]
    pts = sorted(set(pts))
    return mp.quad(h, pts)


def K(a):
    pa0 = 1 / (a * (a + 1) * np.log((a + 1) / a))
    return float(mp.gamma(a) * pa0 / mp.beta(a, 2))


ngrid = np.unique(np.round(np.logspace(np.log10(3), 6, 22)).astype(int))
PN = {a: np.array([float(p_marg(int(m), a)) for m in ngrid]) for a in (1, 2)}
for a in (1, 2):
    for m in (10 ** 3, 10 ** 6):
        r = float(p_marg(m, a)) * m ** (1 + a) * np.log(m) ** 2 / K(a)
        print(f"a = {a}: p(n) n^(1+a) (log n)^2 / K at n = {m:.0e}: {r:.3f}   (K = {K(a):.4f})")

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.2))
ax[0].plot(ns, lk, "o", ms=3.5, color="#3b6a9a", label=r"$\log_{10}\kappa_n$")
xx = np.linspace(5, 165, 2)
ax[0].plot(xx, s * xx + b, "--", lw=0.9, color="0.3", label=rf"fit ${s:.3f}\,n{b:+.2f}$")
ax[0].set_xlabel(r"$n$ (with $k=\lceil n/8\rceil$)")
ax[0].set_ylabel(r"$\log_{10}\kappa_n$")
ax[0].set_title("Cancellation in the residue expansion", fontsize=9)
ax[0].legend(frameon=False, fontsize=8, loc="upper left")
cols = {1: "#3b6a9a", 2: "#5f9e6e"}
mk = {1: "o", 2: "s"}
ls = {1: "--", 2: ":"}
H = []
for a in (1, 2):
    nn = np.logspace(np.log10(3), 6, 200)
    h1, = ax[1].loglog(ngrid, PN[a], mk[a], ms=3, color=cols[a])
    h2, = ax[1].loglog(nn, K(a) * nn ** (-(1 + a)) * np.log(nn) ** -2, ls[a], lw=0.9, color="0.3")
    H.append((h1, h2))
ax[1].set_ylim(1e-22, 1)
ax[1].set_xlabel(r"$n$")
ax[1].set_ylabel(r"$p(n)$")
ax[1].set_title(r"Marginal counts, $b=2$, $c=1$", fontsize=9)
ax[1].legend(H, [r"$a=1$", r"$a=2$"], frameon=False, fontsize=8, loc="upper right", handlelength=3)
fig.tight_layout()
out = os.path.join(_FIGDIR, "rc_tails.pdf")
fig.savefig(out)
print("wrote", out)
