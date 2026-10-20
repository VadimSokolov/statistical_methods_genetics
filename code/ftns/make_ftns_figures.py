"""Regenerate the six figures of "The Horseshoe and the Fundamental Theorem of
Natural Selection" for the book.  Run with:  python -I make_ftns_figures.py

Simulations use the author's samplers in ftns_lib.py (horseshoe,
bivariate_horseshoe).  The design follows the Simulation section of the paper:
L biallelic loci in linkage equilibrium, p ~ U(0.1, 0.9), 20 causal loci with
a ~ N(0, 0.05^2), 10 of which also carry dominance d ~ N(0, 0.03^2), fitness
measured with noise sd 0.10.  The causal architecture and the null genotypes
are drawn from separate random streams so that, in the scaling experiment, the
causal loci stay fixed as L grows.
"""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from multiprocessing import Pool
import ftns_lib as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = str(_FIGDIR)
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
NARROW, WIDE = 3.8, 5.4
N_ITER, BURN = 1200, 400

# ----------------------------------------------------------------------
# design with dominance and exact one generation change
# ----------------------------------------------------------------------
def make_design(seed, L=400, n=300, ncausal=20, ndom=10, sig_e=0.10, Lmax=800):
    rc = np.random.default_rng([seed, 0])
    rn = np.random.default_rng([seed, 1])
    re = np.random.default_rng([seed, 2])
    pc = rc.uniform(0.1, 0.9, ncausal)
    a = rc.standard_normal(ncausal) * 0.05
    d = np.zeros(ncausal); d[:ndom] = rc.standard_normal(ndom) * 0.03
    Gc = rc.binomial(2, pc, size=(n, ncausal)).astype(float)
    pn_all = rn.uniform(0.1, 0.9, Lmax - ncausal)
    Gn_all = rn.binomial(2, pn_all, size=(n, Lmax - ncausal)).astype(float)
    G = np.hstack([Gc, Gn_all[:, :L - ncausal]])
    hc = 2 * pc * (1 - pc)
    alpha_c = a + d * (1 - 2 * pc)
    w = 1.0 + (Gc - 2 * pc) @ a + ((Gc == 1).astype(float) - hc) @ d
    y = w - w.mean() + re.standard_normal(n) * sig_e
    X = G - G.mean(0)
    alpha = np.zeros(L); alpha[:ncausal] = alpha_c
    s2A = float(np.sum(hc * alpha_c ** 2))
    s2D = float(np.sum((hc * d) ** 2))
    # exact one generation change, constant genotypic fitnesses, LE, HWE
    f0, f1, f2 = -2 * pc * a - d * hc, (1 - 2 * pc) * a + d * (1 - hc), 2 * (1 - pc) * a - d * hc
    def wbar(p):
        q = 1 - p
        return 1.0 + np.sum(q * q * f0 + 2 * p * q * f1 + p * p * f2)
    wb = wbar(pc)
    wA = 1.0 + (pc * f2 + (1 - pc) * f1)          # marginal fitness of focal allele
    p1 = pc * wA / wb
    dw_real = wbar(p1) - wb
    return dict(X=X, y=y, alpha=alpha, h=G.var(0), s2A=s2A, s2D=s2D,
                wbar=wb, dw_real=dw_real, dw_partial=s2A / wb)


def marginal(X, y):
    return (X.T @ y) / np.sum(X * X, 0)


def ridge_cv(X, y, k=5, seed=0):
    n = X.shape[0]
    grid = np.logspace(-2, 4, 40)
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    err = np.zeros(len(grid))
    for f in folds:
        tr = np.setdiff1d(idx, f)
        Xt, yt = X[tr], y[tr]
        mx, my = Xt.mean(0), yt.mean()
        U, s, Vt = np.linalg.svd(Xt - mx, full_matrices=False)
        Uy = U.T @ (yt - my)
        for j, lam in enumerate(grid):
            b = Vt.T @ (s / (s ** 2 + lam) * Uy)
            err[j] += np.sum((y[f] - my - (X[f] - mx) @ b) ** 2)
    lam = grid[np.argmin(err)]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ (s / (s ** 2 + lam) * (U.T @ y))


def hs_kappa(X, y, n_iter=N_ITER, burn=BURN, rng=None):
    """ftns_lib.horseshoe, also returning the shrinkage weights
    kappa_l = 1/(1 + n h_l tau^2 lambda_l^2) (prior variance relative to the
    sampling variance sigma^2/(n h_l) of the marginal estimate)."""
    rng = rng or np.random.default_rng(0)
    n, p = X.shape
    nh = np.sum(X * X, 0)
    beta = np.zeros(p); lam2 = np.ones(p); tau2 = 1.0
    nu = np.ones(p); xi = 1.0; sig2 = float(np.var(y))
    keep = np.zeros((n_iter - burn, p)); kap = np.zeros(p)
    for it in range(n_iter):
        D = sig2 * tau2 * lam2
        beta = F._bcm(X, y, D, sig2, rng)
        r = y - X @ beta
        sig2 = 1.0 / rng.gamma((n + p) / 2.0, 1.0 / (0.5 * r @ r + 0.5 * np.sum(beta ** 2 / (tau2 * lam2))))
        lam2 = 1.0 / rng.gamma(1.0, 1.0 / (1.0 / nu + beta ** 2 / (2 * tau2 * sig2)))
        nu = 1.0 / rng.gamma(1.0, 1.0 / (1.0 + 1.0 / lam2))
        tau2 = 1.0 / rng.gamma((p + 1) / 2.0, 1.0 / (1.0 / xi + np.sum(beta ** 2 / lam2) / (2 * sig2)))
        xi = 1.0 / rng.gamma(1.0, 1.0 / (1.0 + 1.0 / tau2))
        if it >= burn:
            keep[it - burn] = beta
            kap += 1.0 / (1.0 + nh * tau2 * lam2)
    return keep, kap / (n_iter - burn)


# ----------------------------------------------------------------------
# workers
# ----------------------------------------------------------------------
def run_main(seed):
    D = make_design(seed)
    X, y, h, al = D["X"], D["y"], D["h"], D["alpha"]
    am = marginal(X, y)
    br = ridge_cv(X, y, seed=seed)
    B, kap = hs_kappa(X, y, rng=np.random.default_rng(seed + 1000))
    m = B.mean(0)
    s2_draws = (B ** 2) @ h
    causal = al != 0
    return dict(seed=seed, s2A=D["s2A"], s2D=D["s2D"], dw_real=D["dw_real"], dw_partial=D["dw_partial"],
                wbar=D["wbar"],
                marg=float(h @ am ** 2), ridge=float(h @ br ** 2), hs_plug=float(h @ m ** 2),
                hs_post=float(s2_draws.mean()),
                mse_marg=float(np.mean((am - al) ** 2)), mse_ridge=float(np.mean((br - al) ** 2)),
                mse_hs=float(np.mean((m - al) ** 2)),
                kappa=kap.tolist(), kap_ratio=np.clip(1 - m / am, 0, 1).tolist(),
                absmarg=np.abs(am).tolist(), causal=causal.tolist())


def run_scaling(args):
    seed, L = args
    D = make_design(seed, L=L)
    X, y, h = D["X"], D["y"], D["h"]
    am = marginal(X, y)
    B = F.horseshoe(X, y, n_iter=N_ITER, burn=BURN, rng=np.random.default_rng(seed + 2000 + L))
    return dict(seed=seed, L=L, s2A=D["s2A"], marg=float(h @ am ** 2), hs_post=float(((B ** 2) @ h).mean()))


def run_coverage(args):
    seed, reg = args
    D = make_design(seed)
    X, y, h = D["X"], D["y"], D["h"]
    B = F.horseshoe(X, y, n_iter=N_ITER, burn=BURN, rng=np.random.default_rng(seed + 3000), regularized=reg)
    s = (B ** 2) @ h
    lo, hi = np.quantile(s, [0.05, 0.95])
    return dict(seed=seed, reg=reg, s2A=D["s2A"], lo=float(lo), hi=float(hi), mean=float(s.mean()))


def run_secondary(seed, L=300, n=250, ncausal=20, r_ab=0.8, rho=0.5, sig=0.10):
    rc = np.random.default_rng([seed, 10])
    p = rc.uniform(0.1, 0.9, L)
    G = rc.binomial(2, p, size=(n, L)).astype(float)
    a = np.zeros(L); b = np.zeros(L)
    za = rc.standard_normal(ncausal); zb = r_ab * za + np.sqrt(1 - r_ab ** 2) * rc.standard_normal(ncausal)
    a[:ncausal] = 0.05 * za; b[:ncausal] = 0.05 * zb
    X = G - G.mean(0)
    h = G.var(0); hp = 2 * p * (1 - p)
    e1 = rc.standard_normal(n); e2 = rho * e1 + np.sqrt(1 - rho ** 2) * rc.standard_normal(n)
    yw = X @ a + sig * e1; yw -= yw.mean()
    yz = X @ b + sig * e2; yz -= yz.mean()
    truth = float(np.sum(hp * a * b))
    am, bm = marginal(X, yw), marginal(X, yz)
    A, Bd = F.bivariate_horseshoe(X, yw, yz, n_iter=N_ITER, burn=BURN, rng=np.random.default_rng(seed + 4000))
    ma, mb = A.mean(0), Bd.mean(0)
    post = float(np.mean(np.sum(A * Bd * h, 1)))
    return dict(seed=seed, truth=truth, marg=float(h @ (am * bm)), hs_plug=float(h @ (ma * mb)), hs_post=post)


def cached(name, fn, tasks, pool):
    path = os.path.join(CACHE, name + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    t = time.time()
    out = pool.map(fn, tasks)
    json.dump(out, open(path, "w"))
    print(f"{name}: {len(tasks)} tasks in {time.time() - t:.0f}s", flush=True)
    return out


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
def style():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
        "font.weight": "normal", "axes.titleweight": "normal", "axes.labelweight": "normal",
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})

# muted colourblind safe (Tol muted)
C = dict(marg="#CC6677", ridge="#DDCC77", hsplug="#88CCEE", hspost="#332288",
         reg="#117733", grey="#777777", black="#222222")


def jitter(k, i, w=0.12, seed=0):
    return i + np.random.default_rng(seed + i).uniform(-w, w, k)


def fig_variance(R):
    names = ["marg", "ridge", "hs_plug", "hs_post"]
    labs = ["marginal", "ridge (CV)", "horseshoe\nplug in", "horseshoe\nposterior"]
    cols = [C["marg"], C["ridge"], C["hsplug"], C["hspost"]]
    truth = np.array([r["s2A"] for r in R])
    fig, ax = plt.subplots(1, 2, figsize=(WIDE, 2.6))
    for i, (k, c) in enumerate(zip(names, cols)):
        v = np.array([r[k] for r in R])
        x = jitter(len(v), i)
        ax[0].scatter(x, v, s=10, color=c, edgecolor="none", alpha=0.9, zorder=3)
        ax[0].hlines(v.mean(), i - 0.25, i + 0.25, color=C["black"], lw=0.9, zorder=4)
        ax[1].scatter(x, v / truth, s=10, color=c, edgecolor="none", alpha=0.9, zorder=3)
        ax[1].hlines((v / truth).mean(), i - 0.25, i + 0.25, color=C["black"], lw=0.9, zorder=4)
    ax[0].scatter(jitter(len(truth), 4), truth, s=10, color=C["grey"], edgecolor="none", zorder=3)
    ax[0].hlines(truth.mean(), 3.75, 4.25, color=C["black"], lw=0.9)
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels(labs + ["truth"], rotation=0, fontsize=7.5)
    ax[0].set_ylabel(r"estimated $\sigma^2_A$")
    ax[1].axhline(1.0, color=C["grey"], ls=":", lw=1.0)
    ax[1].set_yscale("log")
    ax[1].set_yticks([0.25, 0.5, 1, 2, 4]); ax[1].set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    ax[1].minorticks_off()
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labs, fontsize=7.5)
    ax[1].set_ylabel(r"$\hat\sigma^2_A/\sigma^2_A$")
    for a in ax:
        a.set_xlim(-0.5, a.get_xticks()[-1] + 0.5)
        a.tick_params(axis="x", length=0)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(f"{FIG}/ftns_variance.pdf"); plt.close(fig)


def fig_rate(R):
    real = np.array([r["dw_real"] for r in R]); wb = np.array([r["wbar"] for r in R])
    fig, ax = plt.subplots(figsize=(NARROW, 3.0))
    for k, c, lab, mk in [("marg", C["marg"], "marginal", "o"), ("ridge", C["ridge"], "ridge (CV)", "s"),
                          ("hs_post", C["hspost"], "horseshoe posterior", "D")]:
        v = np.array([r[k] for r in R]) / wb
        ax.scatter(real, v, s=13, marker=mk, color=c, edgecolor="none", label=lab, zorder=3)
    lim = [0.004, 0.15]
    ax.plot(lim, lim, color=C["grey"], ls="--", lw=0.9, label="equality")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.008, 0.04); ax.set_ylim(*lim)
    ax.set_xlabel(r"realized $\Delta\bar w$")
    ax.set_ylabel(r"predicted $\hat\sigma^2_A/\bar w$")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/ftns_rate.pdf"); plt.close(fig)


def fig_shrinkage(r):
    kap = np.array(r["kappa"]); am = np.array(r["absmarg"]); cz = np.array(r["causal"])
    fig, ax = plt.subplots(figsize=(NARROW, 2.8))
    ax.scatter(am[~cz], kap[~cz], s=6, color=C["grey"], alpha=0.6, edgecolor="none",
               label=f"null loci (mean {kap[~cz].mean():.2f})")
    ax.scatter(am[cz], kap[cz], s=16, color=C["hspost"], edgecolor="none",
               label=f"nonzero effect (mean {kap[cz].mean():.2f})")
    ax.set_xlabel(r"$|\hat\alpha^{\,\mathrm{marg}}_\ell|$")
    ax.set_ylabel(r"posterior mean of $\kappa_\ell$")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.2)
    fig.tight_layout(); fig.savefig(f"{FIG}/ftns_shrinkage.pdf"); plt.close(fig)


def fig_scaling(S, Ls):
    fig, ax = plt.subplots(figsize=(NARROW, 2.8))
    for k, c, lab, mk, off in [("marg", C["marg"], "marginal plug in", "o", -0.02),
                               ("hs_post", C["hspost"], "horseshoe posterior", "D", 0.02)]:
        mu, se = [], []
        for L in Ls:
            v = np.array([s[k] / s["s2A"] for s in S if s["L"] == L])
            mu.append(v.mean()); se.append(v.std(ddof=1) / np.sqrt(len(v)))
        mu, se = np.array(mu), np.array(se)
        ax.errorbar(np.array(Ls) * (1 + off), mu, yerr=2 * se, color=c, marker=mk, ms=4, capsize=2,
                    lw=1.0, elinewidth=0.8, label=lab)
    ax.axhline(1.0, color=C["grey"], ls=":", lw=1.0)
    ax.set_xscale("log"); ax.set_xticks(Ls); ax.set_xticklabels([str(L) for L in Ls]); ax.minorticks_off()
    ax.set_xlabel(r"number of loci $L$")
    ax.set_ylabel(r"$\hat\sigma^2_A/\sigma^2_A$")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/ftns_scaling.pdf"); plt.close(fig)


def fig_coverage(Cv):
    fig, ax = plt.subplots(1, 2, figsize=(WIDE, 2.8), sharey=True)
    for a, reg, title, col in [(ax[0], False, "horseshoe", C["hspost"]),
                               (ax[1], True, "regularized horseshoe", C["reg"])]:
        rr = sorted([c for c in Cv if c["reg"] == reg], key=lambda c: c["s2A"])
        cov = 0
        for i, c in enumerate(rr):
            hit = c["lo"] <= c["s2A"] <= c["hi"]; cov += hit
            a.vlines(i, c["lo"], c["hi"], color=col if hit else C["black"], lw=0.9 if hit else 1.3,
                     alpha=0.75 if hit else 1.0)
            a.hlines(c["s2A"], i - 0.45, i + 0.45, color=C["marg"], lw=1.0)
        a.set_title(f"{title}: coverage {100 * cov / len(rr):.1f}%", fontsize=9)
        a.set_xlabel("replicate, ordered by true $\\sigma^2_A$")
        a.set_yscale("log"); a.set_xticks([])
    ax[0].set_ylabel(r"90% interval for $\sigma^2_A$")
    fig.tight_layout(w_pad=0.8); fig.savefig(f"{FIG}/ftns_coverage.pdf"); plt.close(fig)


def fig_secondary(Sx):
    truth = np.array([s["truth"] for s in Sx])
    names = ["marg", "hs_plug", "hs_post"]
    labs = ["marginal\nplug in", "bivariate\nhorseshoe\nplug in", "bivariate\nhorseshoe\nposterior"]
    cols = [C["marg"], C["hsplug"], C["hspost"]]
    fig, ax = plt.subplots(figsize=(NARROW, 2.8))
    for i, (k, c) in enumerate(zip(names, cols)):
        v = np.array([s[k] for s in Sx]) / truth
        ax.boxplot(v, positions=[i], widths=0.45, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor=C["grey"], lw=0.7),
                   medianprops=dict(color=C["black"], lw=0.9), whiskerprops=dict(color=C["grey"], lw=0.7),
                   capprops=dict(color=C["grey"], lw=0.7))
        ax.scatter(jitter(len(v), i, 0.1), v, s=11, color=c, edgecolor="none", zorder=3)
    ax.axhline(1.0, color=C["grey"], ls=":", lw=1.0)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5", "1", "2", "4"]); ax.minorticks_off()
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=8); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("estimate / true genetic covariance")
    fig.tight_layout(); fig.savefig(f"{FIG}/ftns_secondary.pdf"); plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    Ls = [100, 200, 400, 800]
    with Pool(2) as pool:
        R = cached("main", run_main, list(range(1, 13)), pool)
        S = cached("scaling", run_scaling, [(s, L) for L in Ls for s in range(201, 207)], pool)
        Cv = cached("coverage", run_coverage, [(s, reg) for reg in (False, True) for s in range(101, 161)], pool)
        Sx = cached("secondary", run_secondary, list(range(301, 313)), pool)
    style()
    fig_variance(R); fig_rate(R); fig_shrinkage(R[0]); fig_scaling(S, Ls); fig_coverage(Cv); fig_secondary(Sx)

    # summary
    A = lambda k, L=R: np.array([r[k] for r in L])
    tr = A("s2A")
    out = {"truth_s2A": tr.mean(), "truth_s2A_se": tr.std(ddof=1) / np.sqrt(len(tr)), "s2D": A("s2D").mean(),
           "dw_real": A("dw_real").mean(), "dw_partial": A("dw_partial").mean(),
           "ratio_real_partial": (A("dw_real") / A("dw_partial")).mean()}
    for k in ["marg", "ridge", "hs_plug", "hs_post"]:
        out[k] = A(k).mean(); out[k + "_ratio"] = (A(k) / tr).mean()
    out["marg_bias"] = (A("marg") - tr).mean()
    for k in ["mse_marg", "mse_ridge", "mse_hs"]:
        out[k] = A(k).mean()
    K = np.concatenate([r["kappa"] for r in R]); Kr = np.concatenate([r["kap_ratio"] for r in R])
    cz = np.concatenate([r["causal"] for r in R]).astype(bool)
    out["kappa_null"], out["kappa_signal"] = K[~cz].mean(), K[cz].mean()
    out["kapratio_null"], out["kapratio_signal"] = Kr[~cz].mean(), Kr[cz].mean()
    out["kappa_rep1_null"] = np.array(R[0]["kappa"])[~np.array(R[0]["causal"])].mean()
    out["kappa_rep1_signal"] = np.array(R[0]["kappa"])[np.array(R[0]["causal"])].mean()
    for L in Ls:
        sl = [s for s in S if s["L"] == L]
        out[f"scale_marg_{L}"] = np.mean([s["marg"] / s["s2A"] for s in sl])
        out[f"scale_hs_{L}"] = np.mean([s["hs_post"] / s["s2A"] for s in sl])
    for reg in (False, True):
        cc = [c for c in Cv if c["reg"] == reg]
        tag = "reg" if reg else "plain"
        out[f"cov_{tag}"] = np.mean([c["lo"] <= c["s2A"] <= c["hi"] for c in cc])
        out[f"below_{tag}"] = np.mean([c["hi"] < c["s2A"] for c in cc])
        out[f"above_{tag}"] = np.mean([c["lo"] > c["s2A"] for c in cc])
        out[f"meanratio_{tag}"] = np.mean([c["mean"] / c["s2A"] for c in cc])
    st = np.array([s["truth"] for s in Sx])
    for k in ["marg", "hs_plug", "hs_post"]:
        out["sec_" + k] = np.mean([s[k] for s in Sx] / st)
        out["sec_" + k + "_pooled"] = np.mean([s[k] for s in Sx]) / st.mean()
    out = {k: float(v) for k, v in out.items()}
    json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"{k:24s} {v:.5g}")
    print(f"total {time.time() - t0:.0f}s")
