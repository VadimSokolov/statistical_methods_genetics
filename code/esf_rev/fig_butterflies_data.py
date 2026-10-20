"""Plot data for the five butterfly figures of the esf chapter (after the censored refit).

butterflies_refit.json caches the summaries quoted in the text but not the curves the
figures need: the full weighted paintbox spectra A_1..A_24, the Ewens posterior curves,
the sampler draws, the joint (alpha, theta) posterior grid and the predictive draws. This
script recomputes exactly those pieces with the grids, priors and random stream of
butterflies_refit.py (same seed, same order of draws), so the figure values agree with
the cached numbers, and it skips the long profile fits (theta hat, n hat and the two
parameter MLE are read from butterflies_refit.json). The likelihood grid is the slow part
(about 7000 coefficients by FFT, run on several processes).

Run: python3 -I code/esf_rev/fig_butterflies_data.py   (a few minutes);
writes fig_butterflies_data.npz next to this file and prints checks against the cache.
"""
import json
import os
import sys
import multiprocessing as mp
import numpy as np
from scipy.special import gammaln

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from butterflies_censored import a, jv, n0, c, K, logw, log_coef  # noqa: E402


def lc_job(args):
    al, n = args
    return log_coef(int(n) - n0, al)


def main():
    RES = json.load(open(os.path.join(_HERE, "butterflies_refit.json")))
    th0, n = RES["ewens"]["theta"], RES["ewens"]["n"]
    ngrid = np.unique(np.round(np.exp(np.linspace(np.log(6400), np.log(40000), 81))).astype(int))
    agrid = np.round(np.linspace(-0.9, 0.5, 29), 4)
    agrid = np.unique(np.concatenate([agrid, np.round(np.linspace(-0.4, 0.15, 56), 4), [0.0]]))
    agrid[agrid == 0] = 0.0
    thg = np.exp(np.linspace(np.log(60), np.log(3000), 1500))
    jobs = [(al, nn) for al in agrid for nn in ngrid]
    with mp.get_context("fork").Pool(min(8, os.cpu_count() or 1)) as pool:
        LC = np.array(pool.map(lc_job, jobs, chunksize=20)).reshape(len(agrid), len(ngrid))

    i = np.arange(1, K)
    def table_ll(al):
        ia = int(np.where(agrid == al)[0][0])
        M = np.full((len(thg), len(ngrid)), -np.inf)
        ok = thg > max(0.0, -al * (K - 1))
        th = thg[ok]
        s = (K - 1) * np.log(th) if al == 0 else np.log(th[:, None] + i[None, :] * al).sum(1)
        cst = (a * logw(jv, al) - gammaln(a + 1)).sum() - gammaln(c + 1)
        for b, nn in enumerate(ngrid):
            M[ok, b] = s - (gammaln(th + nn) - gammaln(th + 1)) + gammaln(nn + 1) + cst + LC[ia, b]
        return M

    # random stream of butterflies_refit.py: estimator simulation first (only its draws matter here)
    rng = np.random.default_rng(20261020)
    thv = float(round(th0))
    for r in range(2000):
        rng.random(n)

    # weighted paintbox, now keeping the whole spectrum j = 1..24
    def gem(t):
        out = []; rem = 1.0
        while rem > 1e-12:
            v = rng.beta(1, t, 4000); w = rem * np.cumprod(np.concatenate([[1], 1 - v[:-1]])) * v
            out.append(w); rem = rem * np.prod(1 - v)
        return np.concatenate(out)
    betas = (0.0, 0.15, 0.30, 0.50)
    spec = []
    for beta in betas:
        st = []
        for r in range(60):
            p = gem(thv); q = p ** (1 + beta); q /= q.sum()
            cnt = rng.multinomial(n, q); cnt = cnt[cnt > 0]
            bc = np.bincount(cnt, minlength=25)
            st.append(np.concatenate([bc[1:25], [(cnt > 24).sum()]]))
        spec.append(np.mean(st, 0))
        m = spec[-1]
        print("weighted beta", beta, "A1 A2 A5 A10 A20 K<=24 K>24", np.round([m[0], m[1], m[4], m[9], m[19], m[:24].sum(), m[24]], 1),
              "cache", np.round(RES["weighted"][str(beta)], 1))
    spec = np.array(spec)

    # Ewens posterior: n integrated (grid) and at n hat (quadrature, sampler)
    M0 = table_ll(0.0)
    lw = M0 - 0.01 * thg[:, None] + np.log(thg)[:, None] + np.log(ngrid)[None, :]
    W = np.exp(lw - lw.max()); W /= W.sum()
    wt = W.sum(1)
    dens_nint = wt / (thg * np.gradient(np.log(thg)))          # density in theta
    tq = np.linspace(60, 300, 200001)
    lpost = K * np.log(tq) + gammaln(tq) - gammaln(tq + n) - 0.01 * tq
    wq = np.exp(lpost - lpost.max()); dens_q = wq / (wq.sum() * (tq[1] - tq[0]))
    print("quadrature mean", round((dens_q * tq).sum() * (tq[1] - tq[0]), 2), "n integrated mean", round((wt * thg).sum(), 2))
    A0, B0 = 1.0, 0.01; th = float(round(th0)); draws = np.empty(200000)
    for it in range(200000):
        eta = rng.beta(th + 1, n); ce = B0 - np.log(eta)
        odds = (A0 + K - 1) / (n * ce); pi = odds / (1 + odds)
        th = rng.gamma(A0 + K if rng.random() < pi else A0 + K - 1, 1 / ce); draws[it] = th
    print("sampler mean", round(draws.mean(), 2), "90%", np.round(np.quantile(draws, [0.05, 0.95]), 2))

    # two parameter posterior: uniform alpha, uniform log theta, flat n
    big = np.array([table_ll(al) + np.log(ngrid)[None, :] for al in agrid])
    big = big + np.log(np.gradient(agrid))[:, None, None]
    P = np.exp(big - big[np.isfinite(big)].max()); P[~np.isfinite(big)] = 0; P /= P.sum()
    Pat = P.sum(2)
    print("P(alpha<0)", round(Pat[agrid < 0].sum(), 5))

    # posterior predictive, second sample of the same size
    D = 4000
    def predictive(model):
        if model == "ewens":
            idx = rng.choice(W.size, D, p=W.ravel()); ti, ni = np.unravel_index(idx, W.shape)
            al = np.zeros(D); th = thg[ti]; nn = ngrid[ni]
        else:
            idx = rng.choice(P.size, D, p=P.ravel()); ia, ti, ni = np.unravel_index(idx, P.shape)
            al = agrid[ia]; th = thg[ti]; nn = ngrid[ni]
        k = np.full(D, float(K)); new = np.zeros(D)
        for s in range(int(nn.max())):
            act = s < nn
            pr = np.clip((th + al * k) / (th + nn + s), 0, 1)
            b = (rng.random(D) < pr) & act
            k += b; new += b
        return new
    pe = predictive("ewens"); pp = predictive("twoparam")
    for name, v in (("Ewens", pe), ("two parameter", pp)):
        print("predictive", name, "mean", round(v.mean(), 2), "90%", np.quantile(v, [0.05, 0.95]), "P(>75)", round((v > 75).mean(), 4))
    print("cache", RES["bayes"])
    np.savez_compressed(os.path.join(_HERE, "fig_butterflies_data.npz"), betas=np.array(betas), spec=spec,
                        thg=thg, dens_nint=dens_nint, tq=tq[::100], dens_q=dens_q[::100], draws=draws,
                        agrid=agrid, Pat=Pat, pe=pe, pp=pp)


if __name__ == "__main__":
    main()
