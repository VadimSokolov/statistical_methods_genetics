"""Negative Binomial regression on the MASS Insurance data (Section rc:sec:app, Table rc:tab:nbreg).

Reproduces, for the Insurance rows of Table rc:tab:nbreg and the text around it (round 2 items
rc_b-02, rc_b-12, rc_b-14):

1. The profile log likelihood of the dispersion alpha for Claims ~ District + Group + Age with
   offset log(Holders) (R default coding: treatment for District, orthogonal polynomial for the
   ordered factors Group and Age). It increases monotonically to its Poisson limit, so there is no
   finite maximum likelihood dispersion. Printed: the gap to the supremum at several alpha, and the
   one sided 95 per cent likelihood bound alpha_L solving sup - l(alpha) = chi2_1(0.90)/2, that is the
   smallest alpha with likelihood ratio at least exp(-1.353) against the Poisson limit, and the
   two sided version with chi2_1(0.95)/2 = 1.921.
2. The posterior of alpha under the three priors PH(0,1,2,1), Ga(0.01,0.01), Ga(2,1), for two
   models that differ only in where the N(0,100 I) prior sits:
     model A, the (alpha, psi) model stated in the chapter: mean alpha exp(psi_i),
              psi_i = o_i + x_i' beta, beta ~ N(0, 100 I);
     model B, the mean parameterization: mean exp(eta_i), eta_i = o_i + x_i' beta, beta ~ N(0, 100 I).
   Writing b for the log mean coefficients, model A is b ~ N(0,100 I) except that the intercept
   has prior N(log alpha, 100); model B has b_0 ~ N(0, 100).
   Each posterior is computed twice: by a mixing MCMC sampler (random walk on t = log alpha with
   the coefficients redrawn jointly from their Laplace approximation given the proposed t, and
   accepted by Metropolis Hastings, plus an independence Metropolis Hastings refresh of b; this is
   exact for the target), 4 chains of 1000 burn in plus 25000 retained, 1e5 retained draws in all;
   and by Laplace integration over b on a grid in t, as a check.
3. 90 per cent posterior predictive bands per policy cell (5 and 95 per cent points of one
   replicate count per retained draw), their mean width and their empirical coverage.
4. The exact posterior mean of alpha under the half horseshoe for models A and B (Laplace grid in
   log alpha): finite under model A, where the N(0,100) prior on the intercept of psi penalises
   very large alpha, and infinite under model B, where the likelihood tends to a positive constant.

Run:  python3 -I code/rc_rev/nbreg_insurance.py   (about eight minutes; output cached in
      code/rc_rev/nbreg_insurance.txt). The model A draws of alpha, the 90 per cent predictive
      band ends and the posterior mean of each cell mean are saved to
      code/rc_rev/nbreg_insurance_draws.npz for the figure script fig_nbreg.py (saving consumes no
      random numbers, so the printed output is unchanged). Data: code/rc_rev/insurance.csv, exported from
      R package MASS (Insurance), integer codes of the factors; checked equal to MASS::Insurance.
"""
import os
import sys
import time
import numpy as np
from scipy.special import gammaln
from scipy.optimize import brentq

_HERE = os.path.dirname(os.path.abspath(__file__))

D = np.genfromtxt(os.path.join(_HERE, "insurance.csv"), delimiter=",", names=True)
y = D["Claims"].astype(float)
o = np.log(D["Holders"].astype(float))
n = len(y)
print(f"Insurance: {n} cells, mean {y.mean():.3f}, variance (divisor n) {y.var():.2f}")


def design():
    P = np.array([[-3, -1, 1, 3], [1, -1, -1, 1], [-1, 3, -3, 1]], float)
    P = P / np.sqrt((P ** 2).sum(1, keepdims=True))
    cols = [np.ones(n)]
    lev = D["District"].astype(int)
    for j in range(2, 5):
        cols.append((lev == j).astype(float))
    for v in ("Group", "Age"):
        lev = D[v].astype(int)
        for r in P:
            cols.append(r[lev - 1])
    return np.column_stack(cols)


X = design()
p = X.shape[1]
PREC = 0.01  # N(0, 100) prior precision


def loglik(alpha, b):
    eta = o + X @ b
    mu = np.exp(eta)
    if np.isinf(alpha):
        return np.sum(y * eta - mu - gammaln(y + 1))
    return np.sum(gammaln(y + alpha) - gammaln(alpha) - gammaln(y + 1)
                  + alpha * np.log(alpha / (alpha + mu)) + y * np.log(mu / (alpha + mu)))


def newton(alpha, b, prec, pm):
    b = b.copy()
    for _ in range(100):
        mu = np.exp(o + X @ b)
        if np.isinf(alpha):
            g = X.T @ (y - mu); W = mu
        else:
            g = X.T @ (alpha * (y - mu) / (alpha + mu)); W = alpha * mu * (y + alpha) / (alpha + mu) ** 2
        g = g - prec * (b - pm)
        H = X.T @ (X * W[:, None]) + prec * np.eye(p)
        s = np.linalg.solve(H, g)
        b = b + s
        if np.max(np.abs(s)) < 1e-10:
            break
    mu = np.exp(o + X @ b)
    W = mu if np.isinf(alpha) else alpha * mu * (y + alpha) / (alpha + mu) ** 2
    H = X.T @ (X * W[:, None]) + prec * np.eye(p)
    return b, H


b_pois, _ = newton(np.inf, np.r_[np.log(y.sum() / np.exp(o).sum()), np.zeros(p - 1)], 0.0, np.zeros(p))
l_sup = loglik(np.inf, b_pois)


def prof(alpha):
    b, _ = newton(alpha, b_pois, 0.0, np.zeros(p))
    return loglik(alpha, b)


print("\n1. Profile likelihood of alpha (no finite maximum)")
mu0 = np.exp(o + X @ b_pois)
print(f"Poisson fit: Pearson chi2/df = {np.sum((y - mu0) ** 2 / mu0) / (n - p):.3f}")
prev = -np.inf
for a in (2, 8.3, 10, 13.1, 20, 20.1, 50, 74.3, 100, 1e3, 1e4, 1e6):
    v = prof(a)
    print(f"  alpha {a:>9g}: l(alpha) - sup = {v - l_sup:9.3f}")
grid = np.exp(np.linspace(np.log(1), np.log(1e6), 400))
vals = np.array([prof(a) for a in grid])
print("  profile increasing on the grid 1..1e6:", bool(np.all(np.diff(vals) > 0)))
for c, lab in ((2.705543 / 2, "one sided 95% (chi2_1 0.90 / 2 = 1.353)"), (3.841459 / 2, "two sided 95% (chi2_1 0.95 / 2 = 1.921)")):
    aL = np.exp(brentq(lambda t: prof(np.exp(t)) - l_sup + c, 0, 20))
    print(f"  likelihood bound alpha_L, {lab}: {aL:.2f}; ratio to 20.1: {aL / 20.1:.2f}")

priors = {
    "PH(0,1,2,1)": lambda a: -np.log(a + 1) - np.log(a + 2),
    "Ga(0.01,0.01)": lambda a: -0.99 * np.log(a) - 0.01 * a,
    "Ga(2,1)": lambda a: np.log(a) - a,
}


def prior_mean(model, t):
    pm = np.zeros(p)
    if model == "A":
        pm[0] = t
    return pm


def logpost(model, lp, t, b):
    a = np.exp(t)
    pm = prior_mean(model, t)
    return loglik(a, b) - 0.5 * PREC * np.sum((b - pm) ** 2) + lp(a) + t


def lap(model, t, bstart):
    pm = prior_mean(model, t)
    bh, H = newton(np.exp(t), bstart, PREC, pm)
    L = np.linalg.cholesky(H)
    return bh, H, L


def logq(b, bh, H, L):
    d = b - bh
    return -0.5 * d @ H @ d + np.sum(np.log(np.diag(L)))


def run_chain(model, lp, rng, t0, nburn=1000, nkeep=25000, step=1.2):
    t = t0
    bh, H, L = lap(model, t, b_pois)
    b = bh.copy()
    cur = logpost(model, lp, t, b)
    out_t = np.empty(nkeep); out_b = np.empty((nkeep, p)); acc = 0
    for it in range(nburn + nkeep):
        # joint move: t' by random walk, b' from Laplace approximation at t'
        t1 = t + step * rng.standard_normal()
        bh1, H1, L1 = lap(model, t1, bh)
        b1 = bh1 + np.linalg.solve(L1.T, rng.standard_normal(p))
        new = logpost(model, lp, t1, b1)
        r = new - cur + logq(b, bh, H, L) - logq(b1, bh1, H1, L1)
        if np.log(rng.uniform()) < r:
            t, b, cur, bh, H, L = t1, b1, new, bh1, H1, L1
            acc += it >= nburn
        # independence refresh of b given t
        b1 = bh + np.linalg.solve(L.T, rng.standard_normal(p))
        new = logpost(model, lp, t, b1)
        if np.log(rng.uniform()) < new - cur + logq(b, bh, H, L) - logq(b1, bh, H, L):
            b, cur = b1, new
        if it >= nburn:
            out_t[it - nburn] = t; out_b[it - nburn] = b
    return out_t, out_b, acc / nkeep


def ess(x):
    x = np.asarray(x) - np.mean(x); m = len(x)
    f = np.fft.rfft(x, 2 * m); ac = np.fft.irfft(f * np.conj(f))[:m]; ac /= ac[0]
    s = 0.0
    for k in range(1, m - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair < 0:
            break
        s += pair
    return m / (1 + 2 * s - 0) if s > 0 else float(m)


def grid_post(model, lp):
    tg = np.linspace(np.log(0.3), np.log(1e9), 3000)
    lm = []; bs = b_pois
    for t in tg:
        bh, H, L = lap(model, t, bs); bs = bh
        lm.append(logpost(model, lp, t, bh) - np.sum(np.log(np.diag(L))))
    w = np.exp(np.array(lm) - max(lm))
    cdf = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(tg))]); cdf /= cdf[-1]
    return [float(np.exp(np.interp(q, cdf, tg))) for q in (0.5, 0.025, 0.975)]


print("\n2./3. Posterior of alpha and 90% predictive bands, 4 chains x 25000 retained draws")
rng = np.random.default_rng(20261020)
SAVE = {"y": y}
for model in ("A", "B"):
    print(f"\nmodel {model}: " + ("(alpha, psi) model, beta ~ N(0,100I) on psi coefficients (as stated in the chapter)" if model == "A" else "mean parameterization, beta ~ N(0,100I) on log mean coefficients"))
    for name, lp in priors.items():
        t0 = time.time()
        ts, bs, accs, esss = [], [], [], []
        for c in range(4):
            tt, bb, ac = run_chain(model, lp, rng, t0=np.log(20.0) + 2.0 * c)
            ts.append(tt); bs.append(bb); accs.append(ac); esss.append(ess(tt))
        T = np.concatenate(ts); B = np.concatenate(bs); A = np.exp(T)
        med, lo, hi = np.median(A), np.quantile(A, 0.025), np.quantile(A, 0.975)
        mu = np.exp(o[None, :] + B @ X.T)
        yrep = rng.negative_binomial(A[:, None], A[:, None] / (A[:, None] + mu))
        q05 = np.quantile(yrep, 0.05, axis=0); q95 = np.quantile(yrep, 0.95, axis=0)
        cov = np.mean((y >= q05) & (y <= q95)); wid = np.mean(q95 - q05)
        if model == "A":
            key = {"PH(0,1,2,1)": "PH", "Ga(0.01,0.01)": "Gdiff", "Ga(2,1)": "G21"}[name]
            SAVE[f"alpha_{key}"] = A.astype(np.float32); SAVE[f"q05_{key}"] = q05; SAVE[f"q95_{key}"] = q95
            SAVE[f"mubar_{key}"] = mu.mean(0)
        g = grid_post(model, lp)
        el = time.time() - t0
        print(f"  {name:14s} MCMC median {med:9.1f}  2.5% {lo:8.1f}  97.5% {hi:10.1f}  mean {A.mean():9.0f} | "
              f"pred coverage {cov:.3f} width {wid:.1f} | ESS(log alpha) {sum(esss):.0f} of 100000, accept {np.mean(accs):.2f}, {el:.0f}s "
              f"| grid check median {g[0]:.1f} ({g[1]:.1f}, {g[2]:.1f})")
        sys.stdout.flush()
np.savez_compressed(os.path.join(_HERE, "nbreg_insurance_draws.npz"), **SAVE)


print("\n4. Posterior mean of alpha under PH(0,1,2,1), Laplace grid in t = log alpha up to t = 400")
print("   (likelihood replaced by its Poisson limit above alpha = 1e7, where it differs by under 1e-4)")
for model in ("A", "B"):
    tg = np.linspace(np.log(0.3), 400.0, 40001)
    lm = np.empty(len(tg)); bs = b_pois
    for i, t in enumerate(tg):
        pm = prior_mean(model, t)
        a = np.exp(t) if t < np.log(1e7) else np.inf
        bh, H = newton(a, bs, PREC, pm); bs = bh
        ll = loglik(a, bh)
        lpa = -np.log(np.exp(t) + 1) - np.log(np.exp(t) + 2) if t < 300 else -2 * t
        lm[i] = ll - 0.5 * PREC * np.sum((bh - pm) ** 2) + lpa + t - 0.5 * np.linalg.slogdet(H)[1]
    lw = lm - lm.max(); w = np.exp(lw)
    Z = np.trapezoid(w, tg)
    lwe = lw + tg; me = np.exp(lwe - lwe.max())
    logmean = np.log(np.trapezoid(me, tg)) + lwe.max() - np.log(Z)
    tpeak = tg[np.argmax(lwe)]
    print(f"  model {model}: posterior mean = {np.exp(logmean):.0f} (log10 {logmean / np.log(10):.2f}); integrand alpha p(alpha) peaks at log10 alpha = {tpeak / np.log(10):.1f}"
          + ("  (no finite mean: the integrand does not decay, value grows with the cutoff)" if model == "B" else ""))
