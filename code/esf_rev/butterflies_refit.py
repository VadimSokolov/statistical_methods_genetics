"""Refit of Corbet's Malayan butterflies with the censored likelihood (esf chapter, esf_a-06).

Uses the likelihood of butterflies_censored.py: 501 species (3306 individuals) in classes
j = 1..24 as tabulated, 119 further species known only to have more than 24 individuals
(Fisher, Corbet and Williams 1943, Table 1: 620 species in all), total sample size n
unknown and profiled out (or integrated out under a flat prior in the Bayesian part).
Reproduces every butterfly number of the chapter after the refit: the Ewens fit and its
intervals, Table butterflies, the estimator simulation, the weighted paintbox table, the
two parameter fits and Table butterflies2, the new species predictions, the posteriors
and the posterior predictive, and the Good and Toulmin example.

Run: python3 -I code/esf_rev/butterflies_refit.py   (about half an hour, serial);
writes butterflies_refit.json and butterflies_refit.txt next to this file.
"""
import json
import os
import sys
import numpy as np
from scipy.special import gammaln, digamma, polygamma
from scipy.optimize import brentq, minimize_scalar

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from butterflies_censored import a, jv, n0, k0, c, K, logw, log_coef  # noqa: E402

OUT = []
def say(*s):
    line = " ".join(str(x) for x in s)
    print(line, flush=True); OUT.append(line)

RES = {}


def theta_part(al, th, n):
    """theta dependent factor log prod_{i<K}(th + i al) - log (th+1)_{n-1}, vectorised in th."""
    th = np.atleast_1d(th).astype(float)
    i = np.arange(1, K)
    if al == 0:
        s = (K - 1) * np.log(th)
    else:
        s = np.log(th[:, None] + i[None, :] * al).sum(1)
    return s - (gammaln(th + n) - gammaln(th + 1))


def const_part(al, n, lc):
    return gammaln(n + 1) + (a * logw(jv, al) - gammaln(a + 1)).sum() + lc - gammaln(c + 1)


def lc_job(args):
    al, n = args
    return log_coef(int(n) - n0, al)


def EA_py(al, th, n, j):
    """exact E A_j of the two parameter law, eq pyEA; alpha = 0 gives eq EAj."""
    j = np.asarray(j, dtype=float)
    return np.exp(gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1)
                  + gammaln(j - al) - gammaln(1 - al)
                  + gammaln(th + al + n - j) - gammaln(th + al)
                  - gammaln(th + n) + gammaln(th + 1))


def EK_py(al, th, n):
    if al == 0:
        return th * (digamma(th + n) - digamma(th))
    return th / al * (np.exp(gammaln(th + al + n) - gammaln(th + al) - gammaln(th + n) + gammaln(th)) - 1)


def new_species(al, th, n, t, k):
    """E[K_{n+t} - K_n | K_n = k]."""
    if al == 0:
        i = np.arange(n, n + t)
        return (th / (th + i)).sum()
    r = np.exp(gammaln(th + n + al + t) - gammaln(th + n + al) - gammaln(th + n + t) + gammaln(th + n))
    return (k + th / al) * (r - 1)


if __name__ == "__main__":
    # ---------------------------------------------------------------- grids
    ngrid = np.unique(np.round(np.exp(np.linspace(np.log(6400), np.log(40000), 81))).astype(int))
    agrid = np.round(np.linspace(-0.9, 0.5, 29), 4)
    agrid = np.unique(np.concatenate([agrid, np.round(np.linspace(-0.4, 0.15, 56), 4), [0.0]]))
    agrid[agrid == 0] = 0.0
    jobs = [(al, n) for al in agrid for n in ngrid]
    LC = np.array([lc_job(jb) for jb in jobs]).reshape(len(agrid), len(ngrid))
    thg = np.exp(np.linspace(np.log(60), np.log(3000), 1500))

    def table_ll(al):
        """log likelihood on (theta grid) x (n grid) for one alpha, -inf where inadmissible."""
        ia = int(np.where(agrid == al)[0][0])
        M = np.full((len(thg), len(ngrid)), -np.inf)
        ok = thg > max(0.0, -al * (K - 1))
        for b, n in enumerate(ngrid):
            M[ok, b] = theta_part(al, thg[ok], n) + const_part(al, n, LC[ia, b])
        return M

    # ------------------------------------------------------------ Ewens fit
    def prof_fixed_n(al, n, lc):
        lo = max(1e-6, -al * (K - 1) + 1e-9)
        r = minimize_scalar(lambda t: -theta_part(al, lo + np.exp(t), n)[0], bounds=(-12, 9),
                            method="bounded", options={"xatol": 1e-10})
        th = lo + np.exp(r.x)
        return theta_part(al, th, n)[0] + const_part(al, n, lc), th

    def fit_alpha(al):
        ia = int(np.where(agrid == al)[0][0]) if al in agrid else None
        if ia is not None:
            vals = [prof_fixed_n(al, n, LC[ia, b])[0] for b, n in enumerate(ngrid)]
            b = int(np.argmax(vals)); lo, hi = ngrid[max(b - 2, 0)], ngrid[min(b + 2, len(ngrid) - 1)]
        else:
            lo, hi = 6400, 40000
        cache = {}
        def f(ln):
            n = int(round(np.exp(ln)))
            if n not in cache:
                cache[n] = prof_fixed_n(al, n, log_coef(n - n0, al))
            return -cache[n][0]
        r = minimize_scalar(f, bounds=(np.log(lo), np.log(hi)), method="bounded", options={"xatol": 2e-5})
        n = int(round(np.exp(r.x)))
        best = min(cache, key=lambda m: -cache[m][0])
        return cache[best][0], cache[best][1], best

    ll0, th0, n0hat = fit_alpha(0.0)
    say("EWENS censored MLE theta", round(th0, 4), "n", n0hat, "loglik", round(ll0, 3))
    # conditional quantities at n = n hat (K = 620 sufficient for theta given n)
    n = n0hat
    EK = lambda t: t * (digamma(t + n) - digamma(t))
    VK = lambda t: EK(t) - t ** 2 * (polygamma(1, t) - polygamma(1, t + n))
    t = K / np.log(n); seq = [t]
    for _ in range(7):
        s = K / t - (digamma(t + n) - digamma(t)); J = K / t ** 2 - (polygamma(1, t) - polygamma(1, t + n))
        t = t + s / J; seq.append(t)
    say("Newton from K/log n:", [round(x, 3) for x in seq], "log n", round(np.log(n), 3), "psi(th)", round(digamma(th0), 3))
    Ic = VK(th0) / th0 ** 2
    say("conditional at n hat: VarK", round(VK(th0), 2), "I", round(Ic, 6), "se", round(Ic ** -0.5, 3),
        "naive se sqrt(th/log n)", round(np.sqrt(th0 / np.log(n)), 3))
    # profile likelihood in theta (n profiled over the grid, refined by interpolation)
    M0 = table_ll(0.0)
    prof = M0.max(1)
    pmax = prof.max()
    inside = thg[prof >= pmax - 1.92]
    say("profile LR 95% interval for theta", round(inside.min(), 2), round(inside.max(), 2), "grid max", round(pmax, 3))
    # curvature of the profile
    sel = (prof >= pmax - 3)
    cf = np.polyfit(thg[sel] - th0, prof[sel], 2)
    se_prof = (-2 * cf[0]) ** -0.5
    say("profile se (quadratic fit within 3 units)", round(se_prof, 3), "Wald", round(th0 - 1.96 * se_prof, 2), round(th0 + 1.96 * se_prof, 2))
    profn = M0.max(0); nin = ngrid[profn >= profn.max() - 1.92]
    say("profile LR 95% interval for n", nin.min(), nin.max())
    RES["ewens"] = dict(theta=th0, n=n0hat, loglik=ll0, newton=seq, se_cond=Ic ** -0.5, se_prof=se_prof,
                        lr_int=[inside.min(), inside.max()], n_int=[int(nin.min()), int(nin.max())])

    # ------------------------------------------------------------ Table butterflies
    EA = EA_py(0.0, th0, n, jv)
    x = n / (n + th0); LS = th0 * x ** jv / jv
    ek = EK(th0)
    say("Ewens means j<=24:", np.round(EA, 2).tolist())
    say("log series j<=24:", np.round(LS, 2).tolist(), "max abs diff", round(np.abs(EA - LS).max(), 4), "x", round(x, 5))
    say("sum E A_j j<=24", round(EA.sum(), 2), "E K_n", round(ek, 2), "expected >24", round(ek - EA.sum(), 2),
        "expected individuals j<=24", round((jv * EA).sum(), 1))
    pear = ((a - EA) ** 2 / EA).sum(); pear_c = pear + (c - (ek - EA.sum())) ** 2 / (ek - EA.sum())
    say("Pearson j<=24", round(pear, 2), "with censored cell", round(pear_c, 2))
    RES["table"] = dict(ewens=EA.tolist(), logseries=LS.tolist(), sum24=EA.sum(), EK=ek, pearson=pear, pearson_c=pear_c)

    # ------------------------------------------------------------ simulation of estimators (Feller coupling)
    rng = np.random.default_rng(20261020)
    thv = float(round(th0)); R = 2000
    def ewens_blocks(thv, n):
        i = np.arange(1, n + 1)
        xi = rng.random(n) < thv / (thv + i - 1)
        pos = np.flatnonzero(xi)
        return np.diff(np.append(pos, n))
    Ks = np.empty(R); mle = np.empty(R); thF = np.empty(R); alt = np.empty(R); kle = np.empty(R)
    EKv = lambda t: t * (digamma(t + n) - digamma(t))
    for r in range(R):
        bl = ewens_blocks(thv, n)
        Ks[r] = len(bl); mle[r] = brentq(lambda u: EKv(u) - Ks[r], 1, 1e5)
        thF[r] = 1 / ((bl * (bl - 1)).sum() / (n * (n - 1))) - 1
        cnt = np.bincount(bl, minlength=25)[1:25]
        alt[r] = ((-1.0) ** (jv + 1) * cnt).sum(); kle[r] = cnt.sum()
    say("sim at theta", thv, "n", n, ": mle mean", round(mle.mean(), 2), "sd", round(mle.std(), 2),
        "| theta_F mean", round(thF.mean(), 1), "sd", round(thF.std(), 1), "var ratio", round(thF.var() / mle.var(), 2))
    say("sim alternating sum j<=24 mean", round(alt.mean(), 2), "sd", round(alt.std(), 2), "P(<=75)", round((alt <= 75).mean(), 4),
        "| species <=24 mean", round(kle.mean(), 1))
    jall = np.arange(1, n + 1); EAall = EA_py(0.0, th0, n, jall); s = (-1.0) ** (jall + 1)
    say("E alt sum all j", round((s * EAall).sum(), 2), "j<=24", round((s * EAall)[:24].sum(), 2),
        "E new species t=n", round(new_species(0, th0, n, n, K), 2), "theta log 2", round(th0 * np.log(2), 2))

    # ------------------------------------------------------------ weighted paintbox at (theta hat, n hat)
    def gem(thv):
        out = []; rem = 1.0
        while rem > 1e-12:
            v = rng.beta(1, thv, 4000); w = rem * np.cumprod(np.concatenate([[1], 1 - v[:-1]])) * v
            out.append(w); rem = rem * np.prod(1 - v)
        return np.concatenate(out)
    rows = {}
    for beta in (0.0, 0.15, 0.30, 0.50):
        st = []
        for r in range(60):
            p = gem(thv); q = p ** (1 + beta); q /= q.sum()
            cnt = rng.multinomial(n, q); cnt = cnt[cnt > 0]
            bc = np.bincount(cnt, minlength=25)
            st.append([bc[1], bc[2], bc[5], bc[10], bc[20], bc[1:25].sum(), (cnt > 24).sum()])
        rows[beta] = np.mean(st, 0)
        say("weighted beta", beta, "A1 A2 A5 A10 A20 K<=24 K>24:", np.round(rows[beta], 1).tolist())
    # Ewens law with the same number of species <= 24 as beta = 0.5
    target = rows[0.5][5]
    k24 = lambda t: EA_py(0.0, t, n, jv).sum()
    tsame = brentq(lambda t: k24(t) - target, 1, 2000)
    say("Ewens theta with same K<=24 as beta=0.5:", round(tsame, 1), "A1", round(EA_py(0, tsame, n, 1), 1), "A2", round(EA_py(0, tsame, n, 2), 1))
    RES["weighted"] = {str(b): v.tolist() for b, v in rows.items()}

    # ------------------------------------------------------------ two parameter profile
    prof_a = []
    for al in agrid:
        ll, th, nh = fit_alpha(al)
        prof_a.append((al, ll, th, nh)); say("alpha", al, "loglik", round(ll, 3), "theta", round(th, 2), "n", nh)
    pa = np.array(prof_a)
    b = int(np.argmax(pa[:, 1]))
    r = minimize_scalar(lambda al: -fit_alpha(al)[0], bounds=(pa[max(b - 1, 0), 0], pa[min(b + 1, len(pa) - 1), 0]),
                        method="bounded", options={"xatol": 1e-4})
    alh = float(r.x); llh, thh, nh = fit_alpha(alh)
    say("TWO PARAMETER MLE alpha", round(alh, 4), "theta", round(thh, 2), "n", nh, "loglik", round(llh, 3),
        "m", round(thh / abs(alh), 1) if alh < 0 else None, "LR", round(2 * (llh - ll0), 2))
    pos = pa[pa[:, 0] >= 0]
    say("restricted alpha>=0 best grid alpha", pos[np.argmax(pos[:, 1]), 0], "loglik", round(pos[:, 1].max(), 3))
    if pos[np.argmax(pos[:, 1]), 0] > 0:
        rr = minimize_scalar(lambda al: -fit_alpha(al)[0], bounds=(0, 0.5), method="bounded", options={"xatol": 1e-4})
        say("restricted refined", rr.x, fit_alpha(rr.x))
    EA2 = EA_py(alh, thh, nh, jv); ek2 = EK_py(alh, thh, nh)
    pear2 = ((a - EA2) ** 2 / EA2).sum()
    pear2c = pear2 + (c - (ek2 - EA2.sum())) ** 2 / (ek2 - EA2.sum())
    say("two parameter means j<=24:", np.round(EA2, 2).tolist())
    say("sum", round(EA2.sum(), 2), "E K", round(ek2, 2), "expected >24", round(ek2 - EA2.sum(), 2),
        "Pearson", round(pear2, 2), "with censored cell", round(pear2c, 2))
    ns_e = new_species(0, th0, n0hat, n0hat, K); ns_p = new_species(alh, thh, nh, nh, K)
    say("new species, second sample of the same size: Ewens", round(ns_e, 2), "two parameter", round(ns_p, 2),
        "(t = n hat of each model)", "unconditional EK(2n)-620 two param", round(EK_py(alh, thh, 2 * nh) - K, 2))
    RES["twoparam"] = dict(alpha=alh, theta=thh, n=nh, loglik=llh, m=(thh / abs(alh) if alh < 0 else None),
                           means=EA2.tolist(), EK=ek2, pearson=pear2, pearson_c=pear2c, profile=pa.tolist(),
                           new_ewens=ns_e, new_twoparam=ns_p)

    # ------------------------------------------------------------ Bayes, Ewens: Gamma(1, 0.01) prior, flat prior on n
    lp = M0 + (-0.01 * thg)[:, None] + np.log(thg)[:, None] * 0  # Gamma(1, 0.01) density in theta
    # grid weights: theta grid is log spaced, so multiply by theta for d theta; n grid log spaced, multiply by n
    lw = lp + np.log(thg)[:, None] + np.log(ngrid)[None, :]
    W = np.exp(lw - lw.max()); W /= W.sum()
    wt = W.sum(1); cdf = np.cumsum(wt)
    q = lambda p: np.interp(p, cdf, thg)
    say("Ewens posterior, n integrated: mean", round((wt * thg).sum(), 2), "median", round(q(0.5), 2),
        "90%", round(q(0.05), 2), round(q(0.95), 2), "sd", round(np.sqrt((wt * thg ** 2).sum() - (wt * thg).sum() ** 2), 2),
        "| n post mean", round((W.sum(0) * ngrid).sum()))
    # at fixed n hat: quadrature and the Beta augmentation sampler
    tq = np.linspace(60, 300, 200001)
    lpost = K * np.log(tq) + gammaln(tq) - gammaln(tq + n) - 0.01 * tq
    wq = np.exp(lpost - lpost.max()); wq /= wq.sum(); cq = np.cumsum(wq)
    say("fixed n hat quadrature: mean", round((wq * tq).sum(), 2), "median", round(np.interp(0.5, cq, tq), 2),
        "90%", round(np.interp(0.05, cq, tq), 2), round(np.interp(0.95, cq, tq), 2),
        "sd", round(np.sqrt((wq * tq ** 2).sum() - (wq * tq).sum() ** 2), 2))
    A0, B0 = 1.0, 0.01; th = float(round(th0)); draws = np.empty(200000)
    for it in range(200000):
        eta = rng.beta(th + 1, n); ce = B0 - np.log(eta)
        odds = (A0 + K - 1) / (n * ce); pi = odds / (1 + odds)
        th = rng.gamma(A0 + K if rng.random() < pi else A0 + K - 1, 1 / ce); draws[it] = th
    say("sampler 2e5 sweeps: mean", round(draws.mean(), 2), "median", round(np.median(draws), 2),
        "90%", np.round(np.quantile(draws, [0.05, 0.95]), 2).tolist(), "lag1 acf", round(np.corrcoef(draws[:-1], draws[1:])[0, 1], 3))
    th = float(round(th0)); Eeta = (th + 1) / (th + 1 + n); ce = B0 - np.log(Eeta)
    pi = ((A0 + K - 1) / (n * ce)) / (1 + (A0 + K - 1) / (n * ce))
    say("one sweep at theta", th, ": E eta", f"{th + 1:.0f}/{th + 1 + n:.0f}", round(Eeta, 5), "c_eta", round(ce, 4),
        "pi", round(pi, 4), "Gamma mean", round(K / ce, 2), "sd", round(np.sqrt(K) / ce, 2))

    # ------------------------------------------------------------ Bayes, two parameter: uniform alpha, uniform log theta, flat n
    big = []
    for al in agrid:
        Ma = table_ll(al)
        big.append(Ma + np.log(ngrid)[None, :])     # log spaced n grid, flat prior on n
    big = np.array(big) + np.log(np.gradient(agrid))[:, None, None]   # uniform on alpha (uneven grid)
    # uniform on log theta: the theta grid is log spaced, so no theta factor
    P = np.exp(big - big[np.isfinite(big)].max()); P[~np.isfinite(big)] = 0; P /= P.sum()
    def summ(vals, w):
        o = np.argsort(vals); cw = np.cumsum(w[o]); v = vals[o]
        return (w * vals).sum(), np.interp(0.05, cw, v), np.interp(0.95, cw, v)
    AL, TH, NN = np.meshgrid(agrid, thg, ngrid, indexing="ij")
    w = P.ravel()
    say("two param posterior alpha", np.round(summ(AL.ravel(), w), 3).tolist(),
        "theta", np.round(summ(TH.ravel(), w), 1).tolist(), "n", np.round(summ(NN.ravel(), w), 0).tolist(),
        "P(alpha<0)", round(w[AL.ravel() < 0].sum(), 5))
    neg = (AL.ravel() < 0) & (w > 0)
    mm = TH.ravel()[neg] / np.abs(AL.ravel()[neg])
    say("m = theta/|alpha| (alpha<0 part)", np.round(summ(mm, w[neg] / w[neg].sum()), 1).tolist())

    # ------------------------------------------------------------ posterior predictive, second sample of the same size
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
            i = nn + s
            pr = np.clip((th + al * k) / (th + i), 0, 1)
            b = (rng.random(D) < pr) & act
            k += b; new += b
        return new
    pe = predictive("ewens"); pp = predictive("twoparam")
    for name, v in (("Ewens", pe), ("two parameter", pp)):
        say("predictive", name, "mean", round(v.mean(), 1), "90%", np.quantile(v, [0.05, 0.95]).tolist(),
            "P(>75)", round((v > 75).mean(), 4), "P(>=75)", round((v >= 75).mean(), 4))
    RES["bayes"] = dict(pred_ewens=[pe.mean(), *np.quantile(pe, [0.05, 0.95])], pred_two=[pp.mean(), *np.quantile(pp, [0.05, 0.95])])
    with open(os.path.join(_HERE, "butterflies_refit.json"), "w") as fh:
        json.dump(RES, fh, indent=1, default=float)
    with open(os.path.join(_HERE, "butterflies_refit.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")
