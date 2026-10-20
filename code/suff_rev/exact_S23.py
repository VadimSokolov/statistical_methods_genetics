"""Exact posterior of theta given S = 23 segregating sites, n = 30 sequences,
prior log uniform on [0.5, 30] (the prior of the ABC comparison). Law of S from
S = sum_{j=1}^{n-1} G_j, G_j geometric on {0,1,...} with success prob j/(theta+j)."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

n, s_obs, LO, HI = 30, 23, 0.5, 30.0

def pmf_S(theta, smax):
    p = np.zeros(smax + 1); p[0] = 1.0
    for j in range(1, n):
        q = theta / (theta + j)
        g = (1 - q) * q ** np.arange(smax + 1)
        p = np.convolve(p, g)[: smax + 1]
    return p

def lik(theta):
    return pmf_S(theta, s_obs)[s_obs]

# prior density on theta: 1/(theta log(HI/LO))
post = lambda t: lik(t) / t
Z = quad(post, LO, HI, limit=200, epsabs=0, epsrel=1e-12)[0]
cdf = lambda x: quad(post, LO, x, limit=200, epsabs=0, epsrel=1e-12)[0] / Z
qs = {a: brentq(lambda x: cdf(x) - a, LO, HI, xtol=1e-12) for a in (0.05, 0.5, 0.95)}
print("quantiles", qs)
print("relative width (q95-q05)/median", (qs[0.95] - qs[0.05]) / qs[0.5])
mean = quad(lambda t: t * post(t), LO, HI, limit=200)[0] / Z
print("posterior mean", mean)
an = sum(1 / j for j in range(1, n))
print("a_n", an, "Watterson", s_obs / an)
# MLE
from scipy.optimize import minimize_scalar
r = minimize_scalar(lambda t: -np.log(lik(t)), bounds=(0.5, 30), method="bounded")
print("MLE", r.x)
print("P(S=23 | theta=8)", lik(8.0))
# prior predictive P(S=23) = acceptance prob at eps=0
print("prior predictive P(S=23)", Z / np.log(HI / LO))
# exact acceptance at eps and exact ABC target quantiles
for eps in (0, 2, 5, 10, 20):
    def pe(t):
        p = pmf_S(t, s_obs + eps)
        return p[max(0, s_obs - eps): s_obs + eps + 1].sum()
    f = lambda t: pe(t) / t
    Ze = quad(f, LO, HI, limit=200, epsrel=1e-11)[0]
    ce = lambda x: quad(f, LO, x, limit=200, epsrel=1e-11)[0] / Ze
    q = [brentq(lambda x: ce(x) - a, LO, HI, xtol=1e-10) for a in (0.05, 0.5, 0.95)]
    print(eps, "acc", Ze / np.log(HI / LO), "median", q[1], "relw", (q[2] - q[0]) / q[1])
