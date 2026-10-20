"""Second set of runs for the infinite alleles example (Sections suff:sec:capacity, suff:sec:calib,
suff:sec:locscale of chapters/suff.tex), with many seeds.

Round 3 review: the three small tables of the capacity, calibration and location scale sections
came from an earlier run whose script and caches were lost. This script reproduces that design
from code/suff/suff_gbc.py and fits every configuration from many seeds.

Data: the training set of suff_gbc.py (rng default_rng(20261017), the first 12000 pairs, n = 100)
and an independent held out sample of M_HOLD pairs drawn from the same stream after the 2500
held out pairs of the main run. Only the training randomness (initialisation, minibatches,
quantile levels) varies across seeds; held out risk uses PRNGKey(5) for every fit, so the u draws
are common to all fits.

Configurations (target log theta, Adam 3e-3, cosine decay, batch 256, as in gbcnet.fit):
    d1_h64_s700, d1_h128_s700, d1_h256_s700   capacity: d = 1, hidden width 64, 128, 256, 700 steps
    d4_h64_s700, d8_h64_s700                  wider bottlenecks, for the run to run comparison
    d1_h64_s2100                              uniform u, 2100 steps
    tail_s2100                                u from Beta(1/2, 1/2), importance weight pi sqrt(u(1-u))
    ls_s2100                                  location scale parametrisation, eta = 0.3, d = 1

Fit mode (one SLURM array task, needs jax and optax; run on Hopper, see slurm/second_runs.sbatch):
    python3 -I second_runs.py fit CONFIG SEED0 NSEED OUTDIR
Summary mode (numpy and scipy only):
    python3 -I second_runs.py summary RESULTDIR     # writes second_runs_summary.json beside this file
Per fit metrics go to OUTDIR/CONFIG_sNNN.json; learned quantiles of every fit to
OUTDIR/CONFIG_sSEED0.npz (plot cache), and the exact posterior quantiles to OUTDIR/exact.npz.
"""
import os, sys, json, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
sys.path.insert(0, str(_ROOT / "code" / "suff"))

M_HOLD = 5000
ETA = 0.3
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
CONFIGS = ["d1_h64_s700", "d1_h128_s700", "d1_h256_s700", "d4_h64_s700", "d8_h64_s700",
           "d1_h64_s2100", "tail_s2100", "ls_s2100"]


def data():
    import numpy as np
    import suff_gbc as E
    E.rng = np.random.default_rng(20261017)
    th_tr, f_tr, m_tr, K_tr, _ = E.simulate(E.M_TRAIN)
    E.simulate(E.M_TEST)                      # the main run's held out set, skipped
    th_te, f_te, m_te, K_te, _ = E.simulate(M_HOLD)
    return E, th_tr, f_tr, m_tr, K_tr, th_te, f_te, m_te, K_te


def exact_quantiles(E, K, n=100):
    import numpy as np
    from scipy.special import gammaln
    import gbcnet as G
    grid = np.geomspace(E.LO, E.HI, 3000)
    logprior = -np.log((grid + 1) * (grid + 2)) + np.log(grid)
    q = np.zeros((K.size, len(LEVELS)))
    for k in np.unique(K):
        ll = k * np.log(grid) + gammaln(grid) - gammaln(grid + n) + logprior
        q[K == k] = G.grid_quantiles(grid, np.exp(ll - ll.max()), LEVELS)[0]
    return q


def make_model(kind, d, hidden):
    """Returns (init(key, feat_dim), H(params, u, feats, mask))."""
    import jax, jax.numpy as jnp
    from jax.scipy.special import ndtri
    import gbcnet as G

    if kind == "plain":
        init = lambda key, fd: G.init_deepsets(key, d, feat_dim=fd, hidden=hidden)
        return init, (lambda p, u, f, m: G.H(p, u, f, m))

    def init(key, fd):
        k1, k2, k3 = jax.random.split(key, 3)
        return {"phi": G._mlp_init(k1, [fd, hidden, hidden, d]),
                "ms": G._mlp_init(k2, [d, hidden, hidden, 2]),
                "c": G._mlp_init(k3, [1 + d, hidden, hidden, 1])}

    def H(p, u, f, m):
        T = G.pooled(p, f, m)
        z = ndtri(jnp.clip(u, 1e-6, 1 - 1e-6))
        ms = G._mlp(p["ms"], T)
        c = G._mlp(p["c"], jnp.concatenate([z[:, None], T], axis=1))[:, 0]
        return ms[:, 0] + jnp.exp(ms[:, 1]) * z + ETA * c
    return init, H


def train(init, Hf, feats, mask, target, steps, key, tail=False, batch=256, lr=3e-3):
    """Same scheme as gbcnet.fit; with tail=True u ~ Beta(1/2,1/2) and the check loss is
    weighted by pi sqrt(u(1-u)), whose mean under Beta(1/2,1/2) is 1 (unbiased for the
    uniform risk)."""
    import jax, jax.numpy as jnp, optax
    import gbcnet as G
    feats = jnp.asarray(feats, jnp.float32)
    mask = jnp.asarray(mask, jnp.float32)
    target = jnp.asarray(target, jnp.float32)
    key, k0 = jax.random.split(key)
    params = init(k0, feats.shape[-1])
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps))
    state = opt.init(params)
    N = target.shape[0]

    def loss_fn(p, f, m, t, u, w):
        return jnp.mean(w * G.check_loss(t - Hf(p, u, f, m), u))

    @jax.jit
    def step(p, s, k):
        ki, ku = jax.random.split(k)
        idx = jax.random.randint(ki, (batch,), 0, N)
        if tail:
            u = jnp.clip(jax.random.beta(ku, 0.5, 0.5, (batch,)), 1e-6, 1 - 1e-6)
            w = jnp.pi * jnp.sqrt(u * (1 - u))
        else:
            u = jax.random.uniform(ku, (batch,))
            w = jnp.ones(batch)
        l, g = jax.value_and_grad(loss_fn)(p, feats[idx], mask[idx], target[idx], u, w)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, l

    for _ in range(steps):
        key, k = jax.random.split(key)
        params, state, _ = step(params, state, k)
    return params


def parse(config):
    if config.startswith("tail"):
        return "plain", 1, 64, 2100, True
    if config.startswith("ls"):
        return "ls", 1, 64, 2100, False
    d, h, s = config.split("_")
    return "plain", int(d[1:]), int(h[1:]), int(s[1:]), False


def fit(config, seed0, nseed, outdir):
    import numpy as np
    import jax, jax.numpy as jnp
    import gbcnet as G
    t0 = time.time()
    E, th_tr, f_tr, m_tr, K_tr, th_te, f_te, m_te, K_te = data()
    y_tr, y_te = np.log(th_tr), np.log(th_te)
    check = {"sum_K_train": int(K_tr.sum()), "sum_K_hold": int(K_te.sum())}
    os.makedirs(outdir, exist_ok=True)
    exq = exact_quantiles(E, K_te)
    if not os.path.exists(os.path.join(outdir, "exact.npz")):
        np.savez(os.path.join(outdir, "exact.npz"), exact_q=exq, K=K_te, theta=th_te)
    kind, d, hidden, steps, tail = parse(config)
    steps = int(os.environ.get("SMOKE_STEPS", steps))   # smoke test only
    init, Hf = make_model(kind, d, hidden)
    fj, mj, yj = (jnp.asarray(a, jnp.float32) for a in (f_te, m_te, y_te))
    Hj = jax.jit(Hf)
    allq, allT = [], []
    for seed in range(seed0, seed0 + nseed):
        key = jax.random.PRNGKey(1000 * seed + 23 + CONFIGS.index(config))
        p = train(init, Hf, f_tr, m_tr, y_tr, steps, key, tail=tail)
        kk, loss = jax.random.PRNGKey(5), 0.0
        for _ in range(6):
            kk, k = jax.random.split(kk)
            u = jax.random.uniform(k, (y_te.size,))
            loss += float(jnp.mean(G.check_loss(yj - Hj(p, u, fj, mj), u))) / 6
        lq = np.exp(np.stack([np.asarray(Hj(p, jnp.full(y_te.size, q, jnp.float32), fj, mj))
                              for q in LEVELS], axis=1))
        rel = (lq - exq) / exq
        out = {"config": config, "seed": seed, "loss": loss,
               "coverage90": float(np.mean((th_te >= lq[:, 0]) & (th_te <= lq[:, 4]))),
               "coverage90_exact": float(np.mean((th_te >= exq[:, 0]) & (th_te <= exq[:, 4]))),
               "median_signed_err_pct": dict(zip(["q05", "q25", "q50", "q75", "q95"],
                                                 (100 * np.median(rel, 0)).tolist())),
               "median_abs_err_pct": dict(zip(["q05", "q25", "q50", "q75", "q95"],
                                              (100 * np.median(np.abs(rel), 0)).tolist())),
               "width_ratio_median": float(np.median((lq[:, 4] - lq[:, 0]) / (exq[:, 4] - exq[:, 0]))),
               **check}
        if d == 1:
            T1 = np.asarray(G.pooled(p, fj, mj))[:, 0]
            from scipy.stats import spearmanr
            out["spearman_T_K"] = float(abs(spearmanr(T1, K_te).correlation))
            allT.append(T1)
        if kind == "ls":
            # monotonicity: dH/dz = exp(s) + eta dc/dz > 0 on a grid of z, every held out sample
            T = G.pooled(p, fj, mj)
            ms = G._mlp(p["ms"], T)
            cz = lambda z: G._mlp(p["c"], jnp.concatenate([z[:, None], T], 1))[:, 0].sum()
            dmin = np.inf
            for z in np.linspace(-3.5, 3.5, 29):
                dc = jax.grad(cz)(jnp.full(y_te.size, z, jnp.float32))
                dmin = min(dmin, float(jnp.min(jnp.exp(ms[:, 1]) + ETA * dc)))
            out["min_dH_dz"] = dmin
        out["elapsed_s"] = time.time() - t0
        out["jax"] = jax.__version__
        allq.append(lq)
        json.dump(out, open(os.path.join(outdir, f"{config}_s{seed:03d}.json"), "w"))
        print(json.dumps(out), flush=True)
    np.savez(os.path.join(outdir, f"{config}_s{seed0:03d}.npz"), learned_q=np.array(allq),
             T1=np.array(allT), seeds=np.arange(seed0, seed0 + nseed))


def summary(resdir):
    import glob
    import numpy as np
    from scipy import stats
    R = {}
    for f in sorted(glob.glob(os.path.join(resdir, "*_s[0-9][0-9][0-9].json"))):
        r = json.load(open(f))
        R.setdefault(r["config"], []).append(r)
    keys = ["loss", "coverage90", "width_ratio_median"]
    out = {"n_seeds": {c: len(v) for c, v in R.items()}, "configs": {}, "diffs": {}}
    chk = {(r["sum_K_train"], r["sum_K_hold"]) for v in R.values() for r in v}
    out["data_checksums"] = sorted(chk)
    ex = {r["coverage90_exact"] for v in R.values() for r in v}
    out["coverage90_exact"] = sorted(ex)
    for c, v in R.items():
        s = {}
        for k in keys + ["spearman_T_K", "min_dH_dz"]:
            x = np.array([r[k] for r in v if k in r])
            if x.size:
                s[k] = {"mean": x.mean(), "sd": x.std(ddof=1), "min": x.min(), "max": x.max(),
                        "median": float(np.median(x))}
        for k in ["q05", "q50", "q95"]:
            x = np.array([r["median_signed_err_pct"][k] for r in v])
            s["signed_err_pct_" + k] = {"mean": x.mean(), "sd": x.std(ddof=1)}
        out["configs"][c] = s

    def welch(a, b, k):
        x = np.array([r[k] for r in R[a]])
        y = np.array([r[k] for r in R[b]])
        dm = x.mean() - y.mean()
        va, vb = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
        se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (x.size - 1) + vb ** 2 / (y.size - 1))
        t = stats.t.ppf(0.975, df)
        return {"diff": dm, "ci95": [dm - t * se, dm + t * se],
                "p_two_sided": float(2 * stats.t.sf(abs(dm / se), df))}

    pairs = [("d1_h64_s700", "d4_h64_s700"), ("d1_h64_s700", "d8_h64_s700"),
             ("d4_h64_s700", "d8_h64_s700"), ("d1_h64_s700", "d1_h128_s700"),
             ("d1_h64_s700", "d1_h256_s700"), ("d1_h128_s700", "d1_h256_s700"),
             ("d1_h64_s700", "d1_h64_s2100"), ("tail_s2100", "d1_h64_s2100"),
             ("ls_s2100", "d1_h64_s2100")]
    for a, b in pairs:
        if a in R and b in R:
            out["diffs"][f"{a} minus {b}"] = {k: welch(a, b, k) for k in keys}
    # location scale fits are bimodal: some seeds never find K_n in the bottleneck. Split them by
    # held out loss at 0.12 (the plain fits all lie below 0.1238, the converged ones below 0.116)
    ok = [r for r in R.get("ls_s2100", []) if r["loss"] < 0.12]
    bad = [r for r in R.get("ls_s2100", []) if r["loss"] >= 0.12]
    def block(v):
        b = {k: {"mean": float(np.mean([r[k] for r in v])), "min": float(np.min([r[k] for r in v])),
                 "max": float(np.max([r[k] for r in v]))}
             for k in keys + ["spearman_T_K", "min_dH_dz"]}
        for q in ["q05", "q50", "q95"]:
            b["signed_err_pct_" + q] = float(np.mean([r["median_signed_err_pct"][q] for r in v]))
        b["seeds"] = [r["seed"] for r in v]
        return b
    if ok:
        out["ls_converged"] = block(ok)
        out["ls_failed"] = block(bad)
        x = np.array([r["coverage90"] for r in ok])
        y = np.array([r["coverage90"] for r in R["d1_h64_s2100"]])
        out["ls_converged_minus_plain2100_coverage"] = float(x.mean() - y.mean())
    # monotone ordering of the five fitted quantile levels on every held out sample
    mono = {}
    for f in sorted(glob.glob(os.path.join(resdir, "*.npz"))):
        if f.endswith("exact.npz"):
            continue
        q = np.load(f)["learned_q"]
        c = os.path.basename(f).rsplit("_s", 1)[0]
        mono[c] = mono.get(c, 0) + int(np.sum(np.any(np.diff(q, axis=2) <= 0, axis=2)))
    out["n_nonmonotone_quantile_rows"] = mono
    js = json.dumps(out, indent=1, default=float)
    open(_HERE / "second_runs_summary.json", "w").write(js)
    print(js)


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    else:
        summary(sys.argv[2])
