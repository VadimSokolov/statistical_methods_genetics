"""Censored likelihood for Corbet's Malayan butterflies (esf chapter, item esf_a-06).

Fisher, Corbet and Williams (1943, Table 1 and p. 43) report 620 species in all, of which
501 species (3306 individuals) were taken between 1 and 24 times.  On p. 43 Corbet says
that "collecting of all individuals seen was not continued after 24 specimens had been
taken": collecting was complete up to 24 specimens of a species and selective after that,
so a count above 24 is only a lower bound for the number seen.  The table therefore stops
at 24 (censoring by the collecting rule, not just grouping), the other 119 species are
known only to have more than 24 individuals, and the total sample size n is not reported.  The data are therefore
(a_1, ..., a_24) together with the event "119 further species, each of size at least 25".

Under the two parameter (Pitman) sampling formula with sample size n the probability of
these data is

  n! prod_{i=1}^{K-1}(theta + i alpha) / (theta+1)_{n-1}
     * prod_{j<=24} w_j^{a_j} / a_j!  *  [u^{n-3306}] G(u)^c / c!,

with K = 620, c = 119, w_j = (1-alpha)_{j-1} / j! and G(u) = sum_{j>=25} w_j u^j.  The
coefficient is computed by exponential tilting: with p_j proportional to w_j r^j (j >= 25)
and S the sum of c independent draws from p, [u^m] G^c = G(r)^c r^{-m} P(S = m), with r
chosen so that E S = m and P(S = m) obtained by FFT.  The Ewens model is alpha = 0.
n is unknown, so it is profiled out.  As a check the Ewens fit is repeated in Fisher's
own Poisson form (n mixed over a negative binomial law, A_j independent Poisson with mean
theta x^j / j, and the 119 censored species Poisson with mean theta sum_{j>24} x^j / j).

Run: python3 -I code/esf_rev/butterflies_censored.py   (about a minute; writes
butterflies_censored.json and prints everything quoted in the chapter).
"""
import json
import os
import numpy as np
from scipy.special import gammaln, digamma, polygamma
from scipy.optimize import brentq, minimize_scalar, minimize

_HERE = os.path.dirname(os.path.abspath(__file__))

a = np.array([118, 74, 44, 24, 29, 22, 20, 19, 20, 15, 12, 14,
              6, 12, 6, 9, 9, 6, 10, 10, 11, 5, 3, 3], dtype=float)
jv = np.arange(1, 25)
n0 = int((a * jv).sum())          # 3306 individuals in classes j <= 24
k0 = int(a.sum())                 # 501
c = 119                           # species with more than 24 individuals
K = k0 + c                        # 620


def logw(j, al):
    """log of w_j = (1-alpha)_{j-1} / j!."""
    return gammaln(j - al) - gammaln(1 - al) - gammaln(j + 1)


def log_coef(m, al):
    """log [u^m] G(u)^c for G(u) = sum_{j>=25} w_j u^j, by tilting and FFT."""
    mu_target = m / c
    J0 = np.arange(25, 25 + 400000)
    LW = logw(J0, al)
    def tilt_stats(lr):
        jj = J0
        lp = LW + lr * jj
        mx = lp.max(); p = np.exp(lp - mx)
        keep = p > 1e-18 * p.sum()
        jj, p = jj[keep], p[keep]
        return jj, p, mx
    def mean_at(lr):
        jj, p, _ = tilt_stats(lr)
        return (jj * p).sum() / p.sum()
    lr = brentq(lambda t: mean_at(t) - mu_target, -2.0, -1e-7, xtol=1e-14)
    jj, p, mx = tilt_stats(lr)
    logG = mx + np.log(p.sum())
    p = p / p.sum()
    L = int(jj.max()) + 1
    N = 1
    while N < c * L + 1:
        N *= 2
    vec = np.zeros(L); vec[jj] = p
    f = np.fft.rfft(vec, N) ** c
    dens = np.fft.irfft(f, N)
    pm = dens[m]
    return c * logG - lr * m + np.log(pm)


def loglik(al, th, n, lc):
    i = np.arange(1, K)
    out = gammaln(n + 1) + np.log(th + i * al).sum() - (gammaln(th + n) - gammaln(th + 1))
    out += (a * logw(jv, al) - gammaln(a + 1)).sum() + lc - gammaln(c + 1)
    return out


def best_theta(al, n):
    """maximise over theta for fixed alpha and n (theta > -alpha (K-1) when alpha < 0)."""
    lo = max(1e-6, -al * (K - 1) + 1e-6)
    r = minimize_scalar(lambda t: -loglik(al, lo + np.exp(t), n, 0.0), bounds=(-10, 12),
                        method="bounded", options={"xatol": 1e-9})
    return lo + np.exp(r.x)


cache = {}
def prof_n(al, n):
    n = int(round(n))
    key = (round(al, 6), n)
    if key not in cache:
        lc = log_coef(n - n0, al)
        th = best_theta(al, n)
        cache[key] = (loglik(al, th, n, lc), th)
    return cache[key]


def profile_alpha(al, grid=None):
    if grid is None:
        grid = np.unique(np.round(np.exp(np.linspace(np.log(n0 + 25 * c + 50), np.log(4e5), 40))))
    vals = [prof_n(al, n)[0] for n in grid]
    b = int(np.argmax(vals))
    lo, hi = grid[max(b - 1, 0)], grid[min(b + 1, len(grid) - 1)]
    r = minimize_scalar(lambda ln: -prof_n(al, np.exp(ln))[0], bounds=(np.log(lo), np.log(hi)),
                        method="bounded", options={"xatol": 1e-5})
    n = int(round(np.exp(r.x)))
    ll, th = prof_n(al, n)
    return ll, th, n


if __name__ == "__main__":
    res = {}
    # Ewens, alpha = 0
    ll0, th0, nh0 = profile_alpha(0.0)
    print("Ewens censored fit: theta", th0, "n", nh0, "loglik", ll0)
    res["ewens"] = dict(theta=th0, n=nh0, loglik=ll0)

    # Poisson (log series) form of the Ewens fit
    def pois_negll(p):
        th, x = np.exp(p[0]), 1 / (1 + np.exp(-p[1]))
        mu = th * x ** jv / jv
        mc = th * (-np.log1p(-x)) - mu.sum()
        return -((a * np.log(mu) - mu).sum() + c * np.log(mc) - mc)
    r = minimize(pois_negll, [np.log(140), 5.0], method="Nelder-Mead",
                 options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    thp, xp = np.exp(r.x[0]), 1 / (1 + np.exp(-r.x[1]))
    print("Poisson log series fit: theta", thp, "x", xp, "implied N", thp * xp / (1 - xp),
          "implied S", thp * -np.log1p(-xp))
    res["logseries"] = dict(theta=thp, x=xp, N=thp * xp / (1 - xp))
    with open(os.path.join(_HERE, "butterflies_censored.json"), "w") as fh:
        json.dump(res, fh, indent=1)
