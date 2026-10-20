"""Table coal:tab:exact and the paragraph after it (Section coal:sec:valS of Chapter coal).

Exact posterior quantiles of theta given S at n = 30 under a Ga(3, scale 3) prior, against a quantile
network trained on N = 2e5 prior predictive pairs (theta, S). S is simulated from the coalescent: the
times T_k ~ Exp(k(k-1)/2), the tree length L = sum_k k T_k, and S | L ~ Poisson(theta L / 2). The network
is the one of hap_train.py (2 x 128 ReLU, Adam, check loss on log theta, level input (1/2) logit u,
decaying learning rate), with input log(1+S) and 40 passes. The exact posterior is computed by the
convolution recursion for P(S = s | theta) on a fine theta grid (Corollary exact).

Prints, for S = 20, 40, 60, 90: exact and network quantiles at 5, 25, 50, 75, 95 percent, the largest
discrepancy at those five levels, the largest discrepancy over the 49 levels 0.02, 0.04, ..., 0.98, the
posterior standard deviation and the ratio of the two. Results for each training seed go to
exact_gbc.json next to this script.

Usage: python3 -I code/coal_rev/exact_gbc.py [SEEDS]    (default SEEDS = 0,1; the table uses seed 0)
About two minutes per seed on a laptop.
"""
import sys, json, numpy as np
from scipy import stats
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent

n = 30; shape, scale = 3.0, 3.0; N = 200000
S_SHOW = [20, 40, 60, 90]
L5 = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
L49 = np.arange(1, 50) * 0.02


def train(Ztr, y, seed, epochs=40, bs=256, width=128):
    """Same network and optimiser as train() in hap_train.py."""
    rng = np.random.default_rng(seed)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Z = (Ztr-mu)/sd
    d = Z.shape[1]+1
    W = [rng.normal(0, np.sqrt(2/d), (d, width)), rng.normal(0, np.sqrt(2/width), (width, width)), rng.normal(0, np.sqrt(1/width), (width, 1))]
    b = [np.zeros(width), np.zeros(width), np.array([y.mean()])]
    params = W + b
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    t = 0; Nn = len(y); steps = epochs*(Nn//bs)
    for ep in range(epochs):
        perm = rng.permutation(Nn)
        for k in range(Nn//bs):
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
    return predict


def exact_quantiles(levels):
    grid = np.linspace(1e-4, 80, 80001)
    smax = max(S_SHOW)
    P = np.zeros((smax+1, grid.size)); P[0] = 1
    for k in range(2, n+1):
        p = (k-1)/(k-1+grid); q = 1-p
        new = np.empty_like(P); new[0] = p*P[0]
        for s in range(1, smax+1):
            new[s] = q*new[s-1]+p*P[s]
        P = new
    prior = stats.gamma.pdf(grid, shape, scale=scale)
    out = {}
    for s in S_SHOW:
        post = P[s]*prior
        c = np.concatenate([[0.0], np.cumsum((post[1:]+post[:-1])/2*np.diff(grid))]); c /= c[-1]
        dens = post/np.trapezoid(post, grid)
        mean = np.trapezoid(grid*dens, grid); sd = np.sqrt(np.trapezoid((grid-mean)**2*dens, grid))
        out[s] = (np.interp(levels, c, grid), sd)
    return out


def simulate(rng):
    th = rng.gamma(shape, scale, N)
    ks = np.arange(2, n+1)
    T = rng.exponential(1.0, (N, n-1))/(ks*(ks-1)/2)
    L = T@ks
    S = rng.poisson(th*L/2)
    return th, S


if __name__ == "__main__":
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1").split(",")]
    allL = np.concatenate([L5, L49])
    ex = exact_quantiles(allL)
    res = {"exact": {s: dict(q5=ex[s][0][:5].round(4).tolist(), sd=round(float(ex[s][1]), 4)) for s in S_SHOW}}
    for seed in seeds:
        rng = np.random.default_rng(1000+seed)
        th, S = simulate(rng)
        pred = train(np.log1p(S[:, None].astype(float)), np.log(th), seed=seed)
        Zq = np.log1p(np.array(S_SHOW, float))[:, None]
        Q5 = np.exp(pred(Zq, L5)); Q49 = np.exp(pred(Zq, L49))
        r = {}
        for i, s in enumerate(S_SHOW):
            e5, e49, sd = ex[s][0][:5], ex[s][0][5:], ex[s][1]
            d5 = np.abs(Q5[i]-e5).max(); d49 = np.abs(Q49[i]-e49).max()
            print("seed %d S=%d exact %s gbc %s max5 %.3f max49 %.3f at %.2f sd %.3f ratio49 %.3f" % (
                seed, s, np.round(e5, 3), np.round(Q5[i], 3), d5, d49, L49[np.argmax(np.abs(Q49[i]-e49))], sd, d49/sd))
            r[s] = dict(gbc5=Q5[i].round(4).tolist(), max5=round(float(d5), 4), max49=round(float(d49), 4), ratio49=round(float(d49/sd), 4))
        m49 = np.mean([r[s]["max49"] for s in S_SHOW])
        print("seed %d mean of max49 %.3f" % (seed, m49))
        res["seed%d" % seed] = r
    json.dump(res, open(_HERE / "exact_gbc.json", "w"), indent=1)
