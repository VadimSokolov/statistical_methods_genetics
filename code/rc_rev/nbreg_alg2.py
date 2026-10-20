"""Algorithm 2 of chapter rc (Negative Binomial regression, half horseshoe dispersion, all blocks
exact) run as stated, on the two data sets of Section rc:sec:app (round 3 item: the effective
sample sizes and timings quoted for Algorithm 2 had no script).

Model (Section rc:sec:nb): n_i ~ NB(alpha, pi_i), psi_i = log{(1 - pi_i)/pi_i} = o_i + x_i' beta,
beta ~ N(0, 100 I), so the mean is alpha exp(psi_i). Offsets o_i: 0 for quine, log(Holders) for
Insurance. Designs as in R: quine Days ~ Eth + Sex + Age + Lrn (treatment coding, data exported
from MASS::quine to code/rc_rev/quine.csv); Insurance as in nbreg_insurance.py.

Samplers (one iteration each):
  alg2    Algorithm 2: l_i ~ CRT(n_i, alpha) (only L = sum l_i is needed); tau = E_1 + E_2,
          E_i ~ Exp(r_i + alpha), r_i = 1, 2 for PH(0,1,2,1); alpha ~ Ga(L + 1, lambda + tau);
          omega_i ~ PG(n_i + alpha, psi_i); beta ~ N(Sigma X'(kappa - Omega o), Sigma),
          kappa_i = (n_i - alpha)/2 (the offset enters through Omega o). Under a Ga(a0, b0) prior
          the tau block is dropped and alpha ~ Ga(L + a0, lambda + b0).
  rwmh    the same sampler with the alpha block replaced by a random walk Metropolis step on
          log alpha targeting the same conditional p(alpha | n, beta), step tuned in burn in to an
          acceptance rate near 0.44.
  slice   the same sampler with the alpha block replaced by a slice sampler (stepping out,
          shrinkage, on log alpha) on the same conditional.
  meanpar mean parameterization (model B of nbreg_insurance.py): mu_i = exp(eta_i),
          eta_i = o_i + x_i' beta, beta ~ N(0, 100 I), pi_i = alpha/(alpha + mu_i); alpha by the
          slice sampler on its conditional given beta, then omega_i ~ PG(n_i + alpha, eta_i - log
          alpha) and beta Gaussian with offset o_i - log alpha.

Polya Gamma draws use the series PG(b, c) = (1/(2 pi^2)) sum_k g_k / ((k - 1/2)^2 + c^2/(4 pi^2)),
g_k ~ Ga(b, 1), truncated at 100 terms with the remainder replaced by its exact mean (the remainder
has relative standard deviation below 1e-3 of the draw).

Effective sample size: initial positive sequence estimator (as in nbreg_insurance.py) on the
retained draws of alpha. Both samplers in a comparison cost about the same per iteration, so the
machine independent comparison is ESS per 2e4 retained draws. Effective draws per second: ESS divided by the wall time of the whole
run, burn in included, single thread, on the machine that ran it (timings are machine dependent).

Outputs: printed (cached in code/rc_rev/nbreg_alg2.txt) and code/rc_rev/nbreg_alg2_draws.npz
(quine alpha draws under the three priors, Insurance traces) for fig_nbreg.py.
Run: python3 -I code/rc_rev/nbreg_alg2.py   (seed 20261020; a few minutes)
"""
import os
import time
import numpy as np
from scipy.special import gammaln

_HERE = os.path.dirname(os.path.abspath(__file__))
PREC = 0.01
KPG = 100
NBURN, NKEEP = 2000, 20000


def quine_data():
    D = np.genfromtxt(os.path.join(_HERE, "quine.csv"), delimiter=",", names=True)
    y = D["Days"].astype(float)
    cols = [np.ones(len(y)), (D["Eth"] == 2) * 1.0, (D["Sex"] == 2) * 1.0]
    cols += [(D["Age"] == j) * 1.0 for j in (2, 3, 4)]
    cols += [(D["Lrn"] == 2) * 1.0]
    return y, np.column_stack(cols), np.zeros(len(y))


def insurance_data():
    D = np.genfromtxt(os.path.join(_HERE, "insurance.csv"), delimiter=",", names=True)
    y = D["Claims"].astype(float)
    n = len(y)
    P = np.array([[-3, -1, 1, 3], [1, -1, -1, 1], [-1, 3, -3, 1]], float)
    P = P / np.sqrt((P ** 2).sum(1, keepdims=True))
    cols = [np.ones(n)]
    lev = D["District"].astype(int)
    cols += [(lev == j).astype(float) for j in range(2, 5)]
    for v in ("Group", "Age"):
        lev = D[v].astype(int)
        cols += [r[lev - 1] for r in P]
    return y, np.column_stack(cols), np.log(D["Holders"].astype(float))


KK = np.arange(1, KPG + 1)
DEN0 = (KK - 0.5) ** 2


def rpg(rng, b, c):
    d = c ** 2 / (4 * np.pi ** 2)
    den = DEN0[None, :] + d[:, None]
    g = rng.gamma(np.repeat(b[:, None], KPG, 1))
    s = (g / den).sum(1)
    ac = np.abs(c)
    mean_tot = np.where(ac > 1e-8, b / (2 * np.maximum(ac, 1e-8)) * np.tanh(ac / 2), b / 4)
    part = b * (1 / den).sum(1) / (2 * np.pi ** 2)
    return s / (2 * np.pi ** 2) + (mean_tot - part)


PRIORS = {"PH(0,1,2,1)": ("PH", None), "Ga(0.01,0.01)": ("Ga", (0.01, 0.01)), "Ga(2,1)": ("Ga", (2.0, 1.0))}


def logprior(a, prior):
    kind, ab = prior
    if kind == "PH":
        return -np.log(a + 1) - np.log(a + 2)
    return (ab[0] - 1) * np.log(a) - ab[1] * a


def logcond_A(a, y, lam, prior):
    # p(alpha | n, beta) in the (alpha, psi) model, omega marginalized: prod [alpha]^{n_i} e^{-lambda alpha} p(alpha)
    return np.sum(gammaln(y + a)) - len(y) * gammaln(a) - lam * a + logprior(a, prior)


def logcond_B(a, y, mu, prior):
    return (np.sum(gammaln(y + a) - gammaln(a) + a * np.log(a / (a + mu)) + y * np.log(mu / (a + mu)))
            + logprior(a, prior))


def slice_logalpha(rng, t, f, w=1.0, m=50):
    fx = f(t) + np.log(rng.uniform())
    lo = t - w * rng.uniform(); hi = lo + w
    j = int(m * rng.uniform()); k = m - 1 - j
    while j > 0 and f(lo) > fx:
        lo -= w; j -= 1
    while k > 0 and f(hi) > fx:
        hi += w; k -= 1
    while True:
        t1 = lo + (hi - lo) * rng.uniform()
        if f(t1) > fx:
            return t1
        if t1 < t:
            lo = t1
        else:
            hi = t1


def ess(x):
    x = np.asarray(x) - np.mean(x); m = len(x)
    f = np.fft.rfft(x, 2 * m); ac = np.fft.irfft(f * np.conj(f))[:m]; ac /= ac[0]
    s = 0.0
    for k in range(1, m - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair < 0:
            break
        s += pair
    return m / (1 + 2 * s) if s > 0 else float(m)


def beta_draw(rng, X, kap, om, off):
    Q = X.T @ (X * om[:, None]) + PREC * np.eye(X.shape[1])
    L = np.linalg.cholesky(Q)
    rhs = X.T @ (kap - om * off)
    m = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
    return m + np.linalg.solve(L.T, rng.standard_normal(X.shape[1]))


def run(sampler, y, X, o, prior, rng, a0, nburn=NBURN, nkeep=NKEEP):
    """Returns retained alpha draws, beta draws, wall time (s), acceptance rate of rwmh."""
    jflat = np.concatenate([np.arange(1, int(v) + 1) for v in y]).astype(float)
    a = a0
    # start beta at the Poisson log linear fit, shifted to the psi scale in the (alpha, psi) model
    b = np.zeros(X.shape[1]); b[0] = np.log(y.sum() / np.exp(o).sum())
    for _ in range(50):
        mu = np.exp(o + X @ b)
        b = b + np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (y - mu))
    if sampler != "meanpar":
        b[0] -= np.log(a)
    A = np.empty(nkeep); Bd = np.empty((nkeep, X.shape[1]))
    step, acc, nacc = 1.0, 0, 0
    t0 = time.time()
    for it in range(nburn + nkeep):
        if sampler == "meanpar":
            mu = np.exp(o + X @ b)
            t = slice_logalpha(rng, np.log(a), lambda t: logcond_B(np.exp(t), y, mu, prior) + t)
            a = np.exp(t)
            off = o - np.log(a)
        else:
            psi = o + X @ b
            lam = np.sum(np.logaddexp(0, psi))
            if sampler == "alg2":
                L = np.count_nonzero(rng.uniform(size=jflat.size) < a / (a + jflat - 1))
                if prior[0] == "PH":
                    tau = rng.exponential(1 / (1 + a)) + rng.exponential(1 / (2 + a))
                    a = rng.gamma(L + 1, 1 / (lam + tau))
                else:
                    a = rng.gamma(L + prior[1][0], 1 / (lam + prior[1][1]))
            elif sampler == "slice":
                a = np.exp(slice_logalpha(rng, np.log(a), lambda t: logcond_A(np.exp(t), y, lam, prior) + t))
            elif sampler == "rwmh":
                t1 = np.log(a) + step * rng.standard_normal()
                r = (logcond_A(np.exp(t1), y, lam, prior) + t1) - (logcond_A(a, y, lam, prior) + np.log(a))
                ok = np.log(rng.uniform()) < r
                if ok:
                    a = np.exp(t1)
                if it < nburn:
                    nacc += ok
                    if (it + 1) % 100 == 0:   # tune towards acceptance 0.44 during burn in only
                        step *= np.exp((nacc / 100 - 0.44) * 2); nacc = 0
                else:
                    acc += ok
            off = o
        eta = off + X @ b
        om = rpg(rng, y + a, eta)
        b = beta_draw(rng, X, (y - a) / 2, om, off)
        if it >= nburn:
            A[it - nburn] = a; Bd[it - nburn] = b
    return A, Bd, time.time() - t0, acc / nkeep


def summarize(tag, A, el, extra=""):
    e = ess(A); el_ = ess(np.log(A))
    print(f"  {tag:34s} median {np.median(A):9.2f} (2.5% {np.quantile(A, .025):8.2f}, 97.5% {np.quantile(A, .975):9.2f})"
          f" | ESS(alpha) {e:7.0f}, ESS(log alpha) {el_:7.0f} of {len(A)} | {el:6.1f} s, {e / el:7.1f} eff/s{extra}")
    return e


def predictive(rng, y, X, o, A, Bd, meanpar=False):
    mu = np.exp(o[None, :] + Bd @ X.T)
    if not meanpar:
        mu = A[:, None] * mu
    yrep = rng.negative_binomial(A[:, None], A[:, None] / (A[:, None] + mu))
    q05 = np.quantile(yrep, 0.05, axis=0); q95 = np.quantile(yrep, 0.95, axis=0)
    return np.mean((y >= q05) & (y <= q95)), np.mean(q95 - q05)


if __name__ == "__main__":
    rng = np.random.default_rng(20261020)
    SAVE = {}
    y, X, o = quine_data()
    print(f"quine: K = {len(y)}, sum n = {y.sum():.0f}, mean {y.mean():.2f}, variance (divisor n) {y.var():.1f}")
    print(f"Table rc:tab:nbreg rows: {NBURN} burn in + 100000 retained iterations of Algorithm 2, start alpha = 1")
    for name, prior in PRIORS.items():
        A, Bd, el, _ = run("alg2", y, X, o, prior, rng, 1.0, nkeep=100000)
        cov, wid = predictive(rng, y, X, o, A, Bd)
        summarize(f"Algorithm 2, {name}", A, el, f" | pred coverage {cov:.3f} width {wid:.1f}")
        SAVE["quine_" + {"PH(0,1,2,1)": "PH", "Ga(0.01,0.01)": "Gdiff", "Ga(2,1)": "G21"}[name]] = A
    PH = PRIORS["PH(0,1,2,1)"]
    print(f"quine, PH(0,1,2,1), runs of {NBURN} burn in + {NKEEP} retained: Algorithm 2 against a tuned random walk Metropolis alpha step, 4 runs each")
    for s in range(4):
        A, _, el, _ = run("alg2", y, X, o, PH, rng, 1.0)
        summarize(f"run {s}: Algorithm 2", A, el)
        A, _, el, acc = run("rwmh", y, X, o, PH, rng, 1.0)
        summarize(f"run {s}: random walk Metropolis", A, el, f" | accept {acc:.2f}")

    y, X, o = insurance_data()
    print(f"\nInsurance: K = {len(y)}, sum n = {y.sum():.0f}; prior PH(0,1,2,1); 4 runs per sampler from alpha0 = 20, 50, 200, 1000")
    for s, a0 in enumerate((20.0, 50.0, 200.0, 1000.0)):
        for smp, lab in (("alg2", "Algorithm 2"), ("slice", "slice alpha step, (alpha, psi)"), ("meanpar", "mean parameterization, slice")):
            A, _, el, _ = run(smp, y, X, o, PH, rng, a0)
            summarize(f"run {s}: {lab}", A, el)
            if s == 0:
                SAVE["ins_" + smp] = A
    np.savez_compressed(os.path.join(_HERE, "nbreg_alg2_draws.npz"), **SAVE)
