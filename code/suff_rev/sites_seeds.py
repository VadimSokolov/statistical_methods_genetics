"""Seed study for Table suff:tab:sites (infinite sites model, n = 30).

Round 2 review item suff_b-08: the reading of the gap between the bottlenecks d = 1 and d = 4
rested on three seeds per configuration. This script refits the configurations of
code/suff/suff_p2.py (S only baseline and deep sets bottlenecks d = 1, 2, 4) with many seeds on
the same simulated training and held out sets (same data seed 20261017, same STEPS, same network,
same held out risk with PRNGKey(5)), seed s using PRNGKey(1000 s + 11) exactly as suff_p2.py does,
so seeds 0, 1, 2 repeat the shipped fits up to library versions. Only the training randomness
(initialisation, minibatches, quantile levels) varies across seeds; the data are fixed, as in the
shipped table.

Fit mode (one SLURM array task, needs jax and optax; run on Hopper, see slurm/sites_seeds.sbatch):
    python3 -I sites_seeds.py fit CONFIG SEED0 NSEED OUTDIR    # CONFIG in S_only, d1, d2, d4
Summary mode (numpy and scipy only, runs on the laptop):
    python3 -I sites_seeds.py summary RESULTDIR                 # writes sites_seeds_summary.json
The cached per seed losses are in sites_seeds_losses.json and the summary in
sites_seeds_summary.json, both beside this script.
"""
import os, sys, json, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
sys.path.insert(0, str(_ROOT / "code" / "suff"))


def fit(config, seed0, nseed, outdir):
    import numpy as np
    import jax
    import gbcnet as G
    import suff_p2 as P
    t0 = time.time()
    rng = np.random.default_rng(20261017)
    th_tr = np.exp(rng.uniform(np.log(P.LO), np.log(P.HI), P.M_TRAIN))
    th_te = np.exp(rng.uniform(np.log(P.LO), np.log(P.HI), P.M_TEST))
    xi_tr, xi_te = P.sim_sfs(th_tr, rng), P.sim_sfs(th_te, rng)
    f_tr, f_te = P.features(xi_tr), P.features(xi_te)
    m_tr, m_te = np.ones(f_tr.shape[:2], np.float32), np.ones(f_te.shape[:2], np.float32)
    y_tr, y_te = np.log(th_tr), np.log(th_te)
    S_tr, S_te = xi_tr.sum(1), xi_te.sum(1)
    s_in = lambda S: np.log1p(S)[:, None].astype(np.float32)
    check = {"sum_S_train": int(S_tr.sum()), "sum_S_test": int(S_te.sum())}
    os.makedirs(outdir, exist_ok=True)
    for seed in range(seed0, seed0 + nseed):
        k1, kt = jax.random.PRNGKey(1000 * seed + 11), jax.random.PRNGKey(5)
        if config == "S_only":
            p = G.fit(s_in(S_tr), None, y_tr, 1, P.STEPS, k1, plain_input=True)
            loss = G.heldout_risk(p, s_in(S_te), None, y_te, kt, plain_input=True)
        else:
            d = int(config[1:])
            p = G.fit(f_tr, m_tr, y_tr, d, P.STEPS, k1)
            loss = G.heldout_risk(p, f_te, m_te, y_te, kt)
        out = {"config": config, "seed": seed, "loss": float(loss), **check,
               "elapsed_s": time.time() - t0, "jax": jax.__version__, "numpy": np.__version__}
        json.dump(out, open(os.path.join(outdir, f"{config}_s{seed:03d}.json"), "w"))
        print(out, flush=True)


def summary(resdir):
    import glob
    import numpy as np
    from scipy import stats
    L = {}
    for f in sorted(glob.glob(os.path.join(resdir, "*_s*.json"))):
        r = json.load(open(f))
        L.setdefault(r["config"], {})[r["seed"]] = r["loss"]
    L = {c: [v[s] for s in sorted(v)] for c, v in L.items()}
    json.dump(L, open(_HERE / "sites_seeds_losses.json", "w"), indent=1)
    out = {"n_seeds": {c: len(v) for c, v in L.items()}, "per_config": {}, "differences": {}}
    for c, v in L.items():
        v = np.array(v)
        out["per_config"][c] = {"mean": v.mean(), "sd": v.std(ddof=1), "min": v.min(),
                                "max": v.max(), "median": float(np.median(v)),
                                "mean_first3": v[:3].mean()}
    rng = np.random.default_rng(0)
    for a, b in [("S_only", "d1"), ("d1", "d2"), ("d2", "d4"), ("d1", "d4")]:
        x, y = np.array(L[a]), np.array(L[b])
        diff = x.mean() - y.mean()
        se = np.sqrt(x.var(ddof=1) / x.size + y.var(ddof=1) / y.size)
        dof = se ** 4 / ((x.var(ddof=1) / x.size) ** 2 / (x.size - 1)
                         + (y.var(ddof=1) / y.size) ** 2 / (y.size - 1))
        tq = stats.t.ppf(0.975, dof)
        boot = np.array([rng.choice(x, x.size).mean() - rng.choice(y, y.size).mean()
                         for _ in range(20000)])
        out["differences"][f"{a}_minus_{b}"] = {
            "diff_of_means": diff, "welch_se": se, "welch_dof": dof,
            "welch_ci95": [diff - tq * se, diff + tq * se],
            "bootstrap_ci95": list(np.quantile(boot, [0.025, 0.975])),
            "welch_p_two_sided": float(stats.ttest_ind(x, y, equal_var=False).pvalue),
            "mannwhitney_p_two_sided": float(stats.mannwhitneyu(x, y).pvalue),
            "frac_pairs_a_gt_b": float(np.mean(x[:, None] > y[None, :]))}
    out = json.loads(json.dumps(out, default=float))
    json.dump(out, open(_HERE / "sites_seeds_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    else:
        summary(sys.argv[2])
