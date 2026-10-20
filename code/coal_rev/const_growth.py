"""Tables coal:tab:const and coal:tab:growth of Chapter coal and the SBC statistics of Section coal:sec:calib.

Round 3 recomputation (the original networks and test sets were lost). Two designs at n = 30:
  const:  theta ~ Ga(4, scale 25), constant population size, 9e4 training and 1.2e4 test replicates.
  growth: theta ~ Ga(4, scale 25), g ~ Unif(0, 20), N(t) = N0 exp(-g t) backwards in time,
          1.2e5 training and 1.2e4 test replicates.
The simulator is vectorised over replicates: coalescent times are drawn in rescaled time
Lambda (rate k(k-1)/2 with k lineages), mapped to real time by t = log(1 + g Lambda)/g, the branch length
l_i subtending i leaves is accumulated while uniformly chosen pairs merge, and xi_i ~ Poisson(theta l_i / 2).
The encoders are the spectrum (input log(1 + xi), 29 cells) and S alone (input log(1 + S)), standardised.
Each parameter gets its own marginal quantile network, the one of hap_train.py (2 x 128 ReLU, Adam, check
loss, level input (1/2) logit u, decaying learning rate), trained for 60 passes; theta is learned on the
log scale and g on the scale logit(g / 20). Scoring is the summarise() of hap_train.py: mean 90% width,
coverage at 90 and 50 percent, and the SBC chi square on nine degrees of freedom from the rank of the true
value among 99 fitted quantiles, in ten bins.

Usage: python3 -I code/coal_rev/const_growth.py SEED
Writes const_growth_seed<SEED>.json (results, unrounded) and const_growth_weights_seed<SEED>.npz (all
network weights and input standardisations) next to this script, and the test sets to
const_growth_test.npz (data seeds are fixed, so the training and test sets do not depend on SEED).
The chapter tables use training seed 0, fixed before any result was seen; seed 1 is a robustness check.
About ten minutes per seed on a laptop.
"""
import sys, json, time, numpy as np
from scipy import stats
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent

n = 30; SHAPE, SCALE = 4.0, 25.0; GMAX = 20.0
LEVELS = np.arange(1, 100) / 100.0


def simulate(M, rng, growth, theta=None):
    th = rng.gamma(SHAPE, SCALE, M) if theta is None else np.full(M, theta)
    g = rng.uniform(0, GMAX, M) if growth else np.zeros(M)
    sizes = np.ones((M, n), dtype=np.int64)
    ell = np.zeros((M, n - 1))
    Lam = np.zeros(M); tprev = np.zeros(M); rows = np.arange(M)
    for k in range(n, 1, -1):
        Lam = Lam + rng.exponential(2.0 / (k * (k - 1)), M)
        safe = np.where(g > 1e-10, g, 1.0)
        t = np.where(g > 1e-10, np.log1p(g * Lam) / safe, Lam)
        dur = t - tprev; tprev = t
        idx = (sizes[:, :k] - 1) + (n - 1) * rows[:, None]
        np.add.at(ell.ravel(), idx.ravel(), np.repeat(dur, k))
        i = rng.integers(0, k, M); j = rng.integers(0, k - 1, M); j = j + (j >= i)
        sizes[rows, i] += sizes[rows, j]
        sizes[rows, j] = sizes[rows, k - 1]
    xi = rng.poisson(th[:, None] * ell / 2)
    return th, g, xi, ell.sum(1)


def train(Ztr, y, seed, epochs=60, bs=256, width=128):
    """Same network and optimiser as train() in hap_train.py; also returns the weights."""
    rng = np.random.default_rng(seed)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Z = (Ztr-mu)/sd
    d = Z.shape[1]+1
    W = [rng.normal(0, np.sqrt(2/d), (d, width)), rng.normal(0, np.sqrt(2/width), (width, width)), rng.normal(0, np.sqrt(1/width), (width, 1))]
    b = [np.zeros(width), np.zeros(width), np.array([y.mean()])]
    params = W + b
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    t = 0; N = len(y); steps = epochs*(N//bs)
    for ep in range(epochs):
        perm = rng.permutation(N)
        for k in range(N//bs):
            idx = perm[k*bs:(k+1)*bs]
            u = rng.uniform(0.005, 0.995, bs)
            x0 = np.hstack([Z[idx], np.log(u[:, None]/(1-u[:, None]))/2])
            h1 = x0@W[0]+b[0]; a1_ = np.maximum(h1, 0)
            h2 = a1_@W[1]+b[1]; a2_ = np.maximum(h2, 0)
            out = (a2_@W[2]).ravel()+b[2]
            r = y[idx]-out
            g = -(u-(r < 0))/bs
            gW2 = a2_.T@g[:, None]; gb2 = np.array([g.sum()])
            d2 = (g[:, None]@W[2].T)*(h2 > 0)
            gW1 = a1_.T@d2; gb1 = d2.sum(0)
            d1 = (d2@W[1].T)*(h1 > 0)
            gW0 = x0.T@d1; gb0 = d1.sum(0)
            grads = [gW0, gW1, gW2, gb0, gb1, gb2]
            t += 1; lr = 1e-3*(0.05**(t/steps))
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9*m[i]+0.1*gr; v[i] = 0.999*v[i]+0.001*gr**2
                p -= lr*(m[i]/(1-0.9**t))/(np.sqrt(v[i]/(1-0.999**t))+1e-8)

    def predict(Zte, u):
        Zs = (Zte-mu)/sd
        outs = []
        for uu in u:
            x0 = np.hstack([Zs, np.full((len(Zs), 1), np.log(uu/(1-uu))/2)])
            a1_ = np.maximum(x0@W[0]+b[0], 0); a2_ = np.maximum(a1_@W[1]+b[1], 0)
            outs.append((a2_@W[2]).ravel()+b[2])
        return np.sort(np.array(outs).T, 1)
    return predict, dict(W0=W[0], W1=W[1], W2=W[2], b0=b[0], b1=b[1], b2=b[2], mu=mu, sd=sd)


def summarise(name, Q, th):
    """As summarise() in hap_train.py."""
    i05, i25, i75, i95 = 4, 24, 74, 94
    w = np.mean(Q[:, i95]-Q[:, i05])
    c90 = np.mean((th > Q[:, i05]) & (th < Q[:, i95])); c50 = np.mean((th > Q[:, i25]) & (th < Q[:, i75]))
    ranks = (Q < th[:, None]).sum(1)
    cnt = np.bincount(np.minimum(ranks//10, 9), minlength=10)
    exp = len(th)*0.10
    chi = ((cnt-exp)**2/exp).sum()
    print("%-22s width %.3f cov90 %.4f cov50 %.4f chi2 %.1f" % (name, w, c90, c50, chi), flush=True)
    return dict(width=float(w), cov90=float(c90), cov50=float(c50), sbc=float(chi))


def feats(xi, kind):
    return np.log1p(xi) if kind == "xi" else np.log1p(xi.sum(1, keepdims=True))


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    res = dict(seed=seed); weights = {}
    # simulator check at g = 0, theta = 8: spectrum means against theta / i
    _, _, xi8, _ = simulate(20000, np.random.default_rng(7), False, theta=8.0)
    res["check_xi_means_theta8"] = {str(i): float(xi8[:, i-1].mean()) for i in (1, 2, 3, 5, 10)}
    print("spectrum means at theta 8 (theory 8/i):", res["check_xi_means_theta8"])
    # prior predictive E[S]: E[theta] E[L]/2, with E[L] from 1e6 simulated trees (exact a1 at g = 0)
    res["ES_const_exact"] = SHAPE*SCALE*sum(1/j for j in range(1, n))
    Ls = np.concatenate([simulate(100000, np.random.default_rng(100+r), True)[3] for r in range(10)])
    res["ES_growth"] = float(SHAPE*SCALE*Ls.mean()/2); res["ES_growth_se"] = float(SHAPE*SCALE*Ls.std()/2/np.sqrt(Ls.size))
    print("E[S] const %.2f growth %.2f (se %.2f)" % (res["ES_const_exact"], res["ES_growth"], res["ES_growth_se"]))
    pw_th = SCALE*(stats.gamma.ppf(0.95, SHAPE)-stats.gamma.ppf(0.05, SHAPE)); res["prior_width_theta"] = float(pw_th)
    res["prior_width_g"] = 0.9*GMAX
    tests = {}
    for design, growth, Ntr, dseed in [("const", False, 90000, 3101), ("growth", True, 120000, 3201)]:
        t0 = time.time()
        th, g, xi, _ = simulate(Ntr, np.random.default_rng(dseed), growth)
        tth, tg, txi, _ = simulate(12000, np.random.default_rng(dseed+1), growth)
        tests[design] = (tth, tg, txi)
        print(design, "simulated in %.1f s, mean S train %.1f test %.1f" % (time.time()-t0, xi.sum(1).mean(), txi.sum(1).mean()), flush=True)
        res[design] = dict(meanS_train=float(xi.sum(1).mean()), meanS_test=float(txi.sum(1).mean()))
        Qp = np.tile(stats.gamma.ppf(LEVELS, SHAPE, scale=SCALE), (len(tth), 1))
        res[design]["prior theta"] = summarise(design+" prior theta", Qp, tth)
        params = [("theta", np.log(th), tth, np.exp)]
        if growth:
            gg = np.clip(g/GMAX, 1e-6, 1-1e-6)
            params.append(("g", np.log(gg/(1-gg)), tg, lambda z: GMAX/(1+np.exp(-z))))
        for kind in ["xi", "S"]:
            for pname, y, truth, inv in params:
                t0 = time.time()
                pred, wts = train(feats(xi, kind), y, seed)
                Q = inv(pred(feats(txi, kind), LEVELS))
                key = "%s %s %s" % (design, kind, pname)
                res[design]["%s %s" % (kind, pname)] = summarise(key, Q, truth)
                print("   train time %.1f" % (time.time()-t0), flush=True)
                for a_, v_ in wts.items():
                    weights["%s_%s_%s_%s" % (design, kind, pname, a_)] = v_
    np.savez_compressed(str(_HERE / ("const_growth_weights_seed%d.npz" % seed)), **weights)
    if not (_HERE / "const_growth_test.npz").exists():
        np.savez_compressed(str(_HERE / "const_growth_test.npz"),
                            **{"%s_%s" % (d, k): v for d, (a_, b_, c_) in tests.items() for k, v in zip(["th", "g", "xi"], [a_, b_, c_.astype(np.int32)])})
    json.dump(res, open(str(_HERE / ("const_growth_seed%d.json" % seed)), "w"), indent=1)
