"""Figure suff_p2: infinite sites worked example, numerical results.

Design as in the Reproducibility section: n = 30 sequences, 30000 training and 4000 held out
pairs, theta log uniform on [0.5, 30], exact coalescent simulation with branch wise Poisson
mutation, per site features (log i, log(1 + xi_i)) over the spectrum i = 1..n-1. Fits: S only
baseline, and deep sets bottlenecks d = 1, 2, 4, each with three seeds (loss panel shows the
mean and range over seeds; the fit carried forward is the seed best on a separate validation set). Exact pi(theta | S) from S = sum_j G_j with G_j
independent geometric, success probability j/(theta + j), on a 300 point geometric grid with S
truncated at 900. ABC: 2e5 prior draws with S generated exactly from the geometric sum, observed
dataset simulated at theta = 8 conditioned to give S = 23, acceptance |S_sim - 23| <= eps.

Run: python3 -I suff_p2.py          (compute and plot)
     python3 -I suff_p2.py plot     (replot from saved results)
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import spearmanr
import jax
import gbcnet as G

t0 = time.time()
outdir = os.path.dirname(os.path.abspath(__file__))
n = 30
M_TRAIN, M_TEST = 30000, 4000
LO, HI = 0.5, 30.0
STEPS = 4000
NSEED = 3
M_VAL = 2000
SMAX = 900
an = np.sum(1.0 / np.arange(1, n))
an2 = np.sum(1.0 / np.arange(1, n) ** 2)


def sim_sfs(theta, rng):
    """Exact coalescent with infinite sites; returns spectrum xi (R, n-1)."""
    R = theta.size
    sizes = np.ones((R, n), int)
    xi = np.zeros((R, n + 1), int)
    rows = np.arange(R)
    for k in range(n, 1, -1):
        T = rng.exponential(1.0 / (k * (k - 1) / 2), R)
        mut = rng.poisson((theta * T / 2)[:, None] * np.ones((1, k)))
        np.add.at(xi, (np.repeat(rows, k), sizes[:, :k].ravel()), mut.ravel())
        i = rng.integers(0, k, R)
        j = rng.integers(0, k - 1, R)
        j = j + (j >= i)
        sizes[rows, i] += sizes[rows, j]
        sizes[rows, j] = sizes[rows, k - 1]
    return xi[:, 1:n]


def features(xi):
    R = xi.shape[0]
    f = np.empty((R, n - 1, 2), np.float32)
    f[:, :, 0] = np.log(np.arange(1, n))[None, :]
    f[:, :, 1] = np.log1p(xi)
    return f


def S_pmf(theta_grid, smax=SMAX, N=4096):
    """pmf of S = sum_{j=1}^{n-1} G_j on 0..smax for each theta, via the generating function."""
    z = np.exp(2j * np.pi * np.arange(N) / N)
    out = np.empty((theta_grid.size, smax + 1))
    for a, th in enumerate(theta_grid):
        pgf = np.ones(N, complex)
        for j in range(1, n):
            p = j / (th + j)
            pgf *= p / (1 - (1 - p) * z)
        pm = np.real(np.fft.fft(pgf)) / N  # coefficients of z^s (conjugate sign)
        out[a] = np.clip(pm[:smax + 1], 0, None)
    return out


def tajima(xi):
    S = xi.sum(1)
    i = np.arange(1, n)
    pi = (i * (n - i) * xi).sum(1) / (n * (n - 1) / 2)
    a1, a2 = an, an2
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n * n + n + 3) / (9 * n * (n - 1))
    c1 = b1 - 1 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / a1 ** 2
    e1, e2 = c1 / a1, c2 / (a1 ** 2 + a2)
    D = (pi - S / a1) / np.sqrt(np.maximum(e1 * S + e2 * S * (S - 1), 1e-12))
    return pi, D


def compute():
    rng = np.random.default_rng(20261017)
    th_tr = np.exp(rng.uniform(np.log(LO), np.log(HI), M_TRAIN))
    th_te = np.exp(rng.uniform(np.log(LO), np.log(HI), M_TEST))
    xi_tr, xi_te = sim_sfs(th_tr, rng), sim_sfs(th_te, rng)
    f_tr, f_te = features(xi_tr), features(xi_te)
    m_tr, m_te = np.ones(f_tr.shape[:2], np.float32), np.ones(f_te.shape[:2], np.float32)
    y_tr, y_te = np.log(th_tr), np.log(th_te)
    S_tr, S_te = xi_tr.sum(1), xi_te.sum(1)
    print("simulated", time.time() - t0, flush=True)

    # validation set, used only to choose which seed's fit is carried forward
    th_va = np.exp(rng.uniform(np.log(LO), np.log(HI), M_VAL))
    xi_va = sim_sfs(th_va, rng)
    f_va = features(xi_va)
    m_va, y_va, S_va = np.ones(f_va.shape[:2], np.float32), np.log(th_va), xi_va.sum(1)

    res = {"loss": {}, "loss_seeds": {}}
    s_in = lambda S: np.log1p(S)[:, None].astype(np.float32)
    fits = {}
    configs = [("S_only", None)] + [(f"d{d}", d) for d in (1, 2, 4)]
    for name, d in configs:
        tl, vl, ps = [], [], []
        for seed in range(NSEED):
            k1 = jax.random.PRNGKey(1000 * seed + 11)
            kt, kv = jax.random.PRNGKey(5), jax.random.PRNGKey(6)
            if d is None:
                p = G.fit(s_in(S_tr), None, y_tr, 1, STEPS, k1, plain_input=True)
                tl.append(G.heldout_risk(p, s_in(S_te), None, y_te, kt, plain_input=True))
                vl.append(G.heldout_risk(p, s_in(S_va), None, y_va, kv, plain_input=True))
            else:
                p = G.fit(f_tr, m_tr, y_tr, d, STEPS, k1)
                tl.append(G.heldout_risk(p, f_te, m_te, y_te, kt))
                vl.append(G.heldout_risk(p, f_va, m_va, y_va, kv))
            ps.append(p)
            print(name, seed, tl[-1], vl[-1], time.time() - t0, flush=True)
        res["loss"][name] = float(np.mean(tl))
        res["loss_seeds"][name] = [float(v) for v in tl]
        if d is not None:
            fits[d] = ps[int(np.argmin(vl))]

    # what the bottleneck coordinates are
    T1 = G.bottleneck(fits[1], f_te, m_te)[:, 0]
    T2 = G.bottleneck(fits[2], f_te, m_te)
    piT, D = tajima(xi_te)
    res["spearman_d1_S"] = float(abs(spearmanr(T1, S_te).correlation))
    res["spearman_d2_S"] = [float(abs(spearmanr(T2[:, c], S_te).correlation)) for c in (0, 1)]
    res["spearman_d2_pi"] = [float(abs(spearmanr(T2[:, c], piT).correlation)) for c in (0, 1)]
    ok = S_te > 1
    X = np.vander(T1[ok], 4)
    rc = []
    for c in (0, 1):
        beta, *_ = np.linalg.lstsq(X, T2[ok, c], rcond=None)
        r = T2[ok, c] - X @ beta
        rc.append(float(np.corrcoef(r, D[ok])[0, 1]))
    res["resid_corr_D"] = rc

    # exact posterior given S, on held out datasets
    grid = np.geomspace(LO, HI, 300)
    pmf = S_pmf(grid)  # (300, 901); log uniform prior is uniform on the geometric grid
    levels = [0.05, 0.5, 0.95]
    nh = 500
    S_h, th_h = S_te[:nh], th_te[:nh]
    eq = G.grid_quantiles(grid, pmf[:, np.minimum(S_h, SMAX)].T, levels)
    gq = np.exp(G.quantiles(fits[4], f_te[:nh], m_te[:nh], levels))
    res["median_relwidth_gen"] = float(np.median((gq[:, 2] - gq[:, 0]) / th_h))
    res["median_relwidth_exactS"] = float(np.median((eq[:, 2] - eq[:, 0]) / th_h))
    res["width_reduction_pct"] = 100 * (1 - res["median_relwidth_gen"] / res["median_relwidth_exactS"])
    res["median_relwidth_gen_over_median"] = float(np.median((gq[:, 2] - gq[:, 0]) / gq[:, 1]))
    res["median_relwidth_exactS_over_median"] = float(np.median((eq[:, 2] - eq[:, 0]) / eq[:, 1]))
    res["coverage_gen"] = float(np.mean((th_h >= gq[:, 0]) & (th_h <= gq[:, 2])))
    res["coverage_exactS"] = float(np.mean((th_h >= eq[:, 0]) & (th_h <= eq[:, 2])))
    print(res, flush=True)

    # ABC comparison at theta = 8 with observed S = 23
    rng2 = np.random.default_rng(8)
    while True:
        xo = sim_sfs(np.array([8.0]), rng2)
        if xo.sum() == 23:
            break
    xo_f = features(xo)
    xo_m = np.ones(xo_f.shape[:2], np.float32)
    thA = np.exp(rng2.uniform(np.log(LO), np.log(HI), 200000))
    p = np.arange(1, n)[None, :] / (thA[:, None] + np.arange(1, n)[None, :])
    SA = (rng2.geometric(p) - 1).sum(1)
    eps_grid = np.arange(0, 21)
    abc = {"eps": [], "acc": [], "ndraw": [], "median": [], "relwidth": []}
    for e in eps_grid:
        acc = np.abs(SA - 23) <= e
        q = np.quantile(thA[acc], levels)
        abc["eps"].append(int(e))
        abc["acc"].append(float(acc.mean()))
        abc["ndraw"].append(int(acc.sum()))
        abc["median"].append(float(q[1]))
        abc["relwidth"].append(float((q[2] - q[0]) / q[1]))
    res["abc"] = abc
    e23 = G.grid_quantiles(grid, pmf[:, 23][None, :], levels)[0]
    res["exactS_obs"] = {"median": float(e23[1]), "relwidth": float((e23[2] - e23[0]) / e23[1])}
    go = np.exp(G.quantiles(fits[4], xo_f, xo_m, levels))[0]
    res["gen_obs"] = {"median": float(go[1]), "relwidth": float((go[2] - go[0]) / go[1])}
    res["runtime_s"] = time.time() - t0
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(outdir, "suff_p2_results.json"), "w"), indent=1)
    np.savez(os.path.join(outdir, "suff_p2_plotdata.npz"), T2=T2, logth=y_te)


def plot():
    res = json.load(open(os.path.join(outdir, "suff_p2_results.json")))
    Dd = np.load(os.path.join(outdir, "suff_p2_plotdata.npz"))
    T2, logth = Dd["T2"], Dd["logth"]
    plt = G.style()
    C = G.COL
    fig, ax = plt.subplots(1, 3, figsize=(5.4, 2.1),
                           gridspec_kw={"width_ratios": [1, 1.15, 1.05]})

    a = ax[0]
    labs = ["$S$", "1", "2", "4"]
    vals = [res["loss"][k] for k in ("S_only", "d1", "d2", "d4")]
    sd = [res["loss_seeds"][k] for k in ("S_only", "d1", "d2", "d4")]
    err = np.array([[v - min(z) for v, z in zip(vals, sd)], [max(z) - v for v, z in zip(vals, sd)]])
    a.errorbar(range(1, 4), vals[1:], yerr=err[:, 1:], color=C["blue"], lw=0.8, marker="o",
               ms=3.5, capsize=2, elinewidth=0.6)
    a.errorbar([0], vals[:1], yerr=err[:, :1], color=C["rose"], marker="s", ms=3.5, lw=0,
               capsize=2, elinewidth=0.6)
    a.set_xticks(range(4))
    a.set_xticklabels(labs)
    a.set_xlim(-0.5, 3.5)
    lo = min(min(z) for z in sd)
    hi = max(max(z) for z in sd)
    pad = 0.15 * (hi - lo)
    a.set_ylim(lo - pad, hi + pad)
    a.set_xlabel("summary or bottleneck $d$", labelpad=1)
    a.set_ylabel("held out check loss")

    a = ax[1]
    sel = np.random.default_rng(2).choice(T2.shape[0], 2000, replace=False)
    sc = a.scatter(T2[sel, 0], T2[sel, 1], c=logth[sel], s=2, lw=0, cmap="viridis",
                   rasterized=True)
    a.set_xlabel("$T_1$")
    a.set_ylabel("$T_2$", labelpad=1)
    cb = fig.colorbar(sc, ax=a, fraction=0.07, pad=0.03)
    cb.set_label(r"$\log\theta$", labelpad=1)
    cb.ax.tick_params(labelsize=7, width=0.5, length=2)
    cb.outline.set_linewidth(0.4)

    a = ax[2]
    ab = res["abc"]
    a.plot(ab["eps"], ab["relwidth"], color=C["blue"], lw=0.9, marker="o", ms=2,
           label="ABC on $S$")
    a.axhline(res["exactS_obs"]["relwidth"], color=C["rose"], lw=0.8, ls="--",
              label=r"exact given $S$")
    a.axhline(res["gen_obs"]["relwidth"], color=C["teal"], lw=0.8, ls=":",
              label="full spectrum")
    a.set_xlabel(r"tolerance $\epsilon$")
    a.set_ylabel(r"90% width / median")
    a.set_ylim(0, max(ab["relwidth"]) * 1.3)
    a.legend(frameon=False, loc="upper left", fontsize=7, handlelength=1.6,
             borderaxespad=0.1)

    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(os.path.join(G.FIGDIR, "suff_p2.pdf"), dpi=300)
    print("done", time.time() - t0)


if __name__ == "__main__":
    if "plot" not in sys.argv[1:]:
        compute()
    plot()
