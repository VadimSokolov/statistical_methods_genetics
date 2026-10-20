"""Figure suff_gbc: infinite alleles worked example.

Does the generative quantile method rediscover Ewens' sufficiency theorem (K_n sufficient)?
Design as in the Reproducibility section: n = 100 genes, 12000 training and 2500 held out
pairs, theta from the Pochhammer prior p(theta) propto 1/((theta+1)(theta+2)) truncated to
[0.2, 25], exact partitions by the Feller coupling, per block features (log n_i, n_i/n),
deep sets with 64 hidden units, SiLU, Adam 3e-3 with cosine decay, batch 256, 700 steps,
target log theta. Exact posterior given K_n on a 3000 point geometric grid.

Run: python3 -I suff_gbc.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import gammaln
from scipy.stats import spearmanr
import jax
import gbcnet as G

t0 = time.time()
rng = np.random.default_rng(20261017)
n = 100
M_TRAIN, M_TEST = 12000, 2500
LO, HI = 0.2, 25.0
STEPS = 700
outdir = os.path.dirname(os.path.abspath(__file__))


def prior_draw(size):
    g = lambda t: np.log((t + 1) / (t + 2))
    v = rng.uniform(g(LO), g(HI), size)
    e = np.exp(v)
    return (2 * e - 1) / (1 - e)


def feller(theta):
    i = np.arange(1, n + 1)
    xi = rng.random(n) < theta / (theta + i - 1)
    xi[0] = True
    pos = np.flatnonzero(xi)
    return np.diff(np.append(pos, n))


def simulate(M):
    th = prior_draw(M)
    feats = np.zeros((M, n, 2), np.float32)
    mask = np.zeros((M, n), np.float32)
    K = np.zeros(M, int)
    mx = np.zeros(M)
    for m in range(M):
        b = feller(th[m])
        k = len(b)
        feats[m, :k, 0] = np.log(b)
        feats[m, :k, 1] = b / n
        mask[m, :k] = 1
        K[m] = k
        mx[m] = b.max() / n
    return th, feats, mask, K, mx


def compute():
    global key
    th_tr, f_tr, m_tr, K_tr, mx_tr = simulate(M_TRAIN)
    th_te, f_te, m_te, K_te, mx_te = simulate(M_TEST)
    y_tr, y_te = np.log(th_tr), np.log(th_te)
    print("simulated", time.time() - t0, flush=True)

    key = jax.random.PRNGKey(7)
    res = {"loss": {}}
    fits = {}
    for d in (1, 4, 8):
        key, k1, k2 = jax.random.split(key, 3)
        p = G.fit(f_tr, m_tr, y_tr, d, STEPS, k1)
        res["loss"][f"d{d}"] = G.heldout_risk(p, f_te, m_te, y_te, k2)
        fits[d] = p
        print("d", d, res["loss"][f"d{d}"], time.time() - t0, flush=True)

    # misspecified model: only the largest allele frequency
    key, k1, k2 = jax.random.split(key, 3)
    pmax = G.fit(mx_tr[:, None], None, y_tr, 1, STEPS, k1, plain_input=True)
    res["loss"]["maxfreq"] = G.heldout_risk(pmax, mx_te[:, None], None, y_te, k2, plain_input=True)
    print("maxfreq", res["loss"]["maxfreq"], flush=True)
    res["excess_maxfreq_over_d8_pct"] = 100 * (res["loss"]["maxfreq"] / res["loss"]["d8"] - 1)
    res["excess_maxfreq_over_d1_pct"] = 100 * (res["loss"]["maxfreq"] / res["loss"]["d1"] - 1)

    # learned scalar against K_n
    T1 = G.bottleneck(fits[1], f_te, m_te)[:, 0]
    rho = spearmanr(T1, K_te).correlation
    res["spearman_T_K"] = float(abs(rho))
    print("spearman", rho, flush=True)

    # exact posterior given K_n
    grid = np.geomspace(LO, HI, 3000)
    dlog = np.log(grid)  # geometric grid: weight by theta for d theta
    logprior = -np.log((grid + 1) * (grid + 2)) + np.log(grid)
    levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    exact_q = np.zeros((M_TEST, 5))
    for k in np.unique(K_te):
        ll = k * np.log(grid) + gammaln(grid) - gammaln(grid + n) + logprior
        w = np.exp(ll - ll.max())
        exact_q[K_te == k] = G.grid_quantiles(grid, w, levels)[0]
    learned_q = np.exp(G.quantiles(fits[1], f_te, m_te, levels))
    relerr = np.median(np.abs(learned_q - exact_q) / exact_q, axis=0)
    sgnerr = np.median((learned_q - exact_q) / exact_q, axis=0)
    res["median_rel_err_pct"] = dict(zip(["q05", "q25", "q50", "q75", "q95"], (100 * relerr).tolist()))
    res["median_signed_err_pct"] = dict(zip(["q05", "q25", "q50", "q75", "q95"], (100 * sgnerr).tolist()))
    res["coverage90_learned"] = float(np.mean((th_te >= learned_q[:, 0]) & (th_te <= learned_q[:, 4])))
    res["coverage90_exact"] = float(np.mean((th_te >= exact_q[:, 0]) & (th_te <= exact_q[:, 4])))
    res["width_ratio_median"] = float(np.median((learned_q[:, 4] - learned_q[:, 0]) /
                                                (exact_q[:, 4] - exact_q[:, 0])))
    res["runtime_s"] = time.time() - t0
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(outdir, "suff_gbc_results.json"), "w"), indent=1)

    np.savez(os.path.join(outdir, "suff_gbc_plotdata.npz"), K_te=K_te, T1=T1,
             exact_q=exact_q, learned_q=learned_q)
    return res


def plot():
    res = json.load(open(os.path.join(outdir, "suff_gbc_results.json")))
    D = np.load(os.path.join(outdir, "suff_gbc_plotdata.npz"))
    K_te, T1, exact_q, learned_q = D["K_te"], D["T1"], D["exact_q"], D["learned_q"]
    rho = res["spearman_T_K"]
    prng = np.random.default_rng(1)
    # figure
    plt = G.style()
    C = G.COL
    fig, ax = plt.subplots(1, 3, figsize=(5.4, 2.05))

    a = ax[0]
    labs = ["1", "4", "8", "max\nfreq"]
    vals = [res["loss"]["d1"], res["loss"]["d4"], res["loss"]["d8"], res["loss"]["maxfreq"]]
    cols = [C["blue"]] * 3 + [C["rose"]]
    a.bar(range(4), vals, color=cols, width=0.6)
    a.set_xticks(range(4))
    a.set_xticklabels(labs)
    a.set_xlabel("bottleneck $d$", labelpad=1)
    a.set_ylim(0, 0.2)
    a.set_ylabel("held out check loss")
    a = ax[1]
    sel = prng.choice(M_TEST, 1200, replace=False)
    jit = prng.uniform(-0.25, 0.25, sel.size)
    a.scatter(K_te[sel] + jit, T1[sel], s=2, color=C["teal"], alpha=0.5, lw=0, rasterized=True)
    a.set_xlabel("$K_n$")
    a.set_ylabel("learned statistic $T_1$")
    a.text(0.97, 0.97, f"Spearman {abs(rho):.3f}", transform=a.transAxes,
           ha="right", va="top", fontsize=8)

    a = ax[2]
    sel = prng.choice(M_TEST, 300, replace=False)
    for j, (lv, c) in enumerate(zip([0, 2, 4], [C["cyan"], C["blue"], C["rose"]])):
        a.scatter(exact_q[sel, lv], learned_q[sel, lv], s=3, color=c, lw=0, alpha=0.8,
                  label=["5%", "50%", "95%"][j], rasterized=True)
    lim = [LO, HI * 1.2]
    a.plot(lim, lim, color=C["grey"], lw=0.7)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(lim)
    a.set_ylim(lim)
    tk = [0.5, 1, 5, 20]
    a.set_xticks(tk)
    a.set_xticklabels([f"{t:g}" for t in tk])
    a.set_yticks(tk)
    a.set_yticklabels([f"{t:g}" for t in tk])
    a.minorticks_off()
    a.set_xlabel(r"exact quantile")
    a.set_ylabel(r"learned quantile")
    a.legend(frameon=False, loc="upper left", handletextpad=0.1, markerscale=2.5, borderaxespad=0.1)

    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(G.FIGDIR, "suff_gbc.pdf"), dpi=300)
    print("done", time.time() - t0)


if __name__ == "__main__":
    if "plot" not in sys.argv[1:]:
        compute()
    plot()
