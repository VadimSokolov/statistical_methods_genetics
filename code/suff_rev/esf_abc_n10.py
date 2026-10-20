"""Toy rejection ABC against the exact posterior, infinite alleles model, n = 10.
Prior: Pochhammer p(theta) = 1/{log 2 (theta+1)(theta+2)} on (0, inf).
Observed allele configuration: block sizes (5,2,2,1), so K = 4, largest block 5."""
import numpy as np
from math import lgamma, log, factorial
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaln
n = 10
obs = (5, 2, 2, 1)
def partitions(m, maxp=None):
    if maxp is None: maxp = m
    if m == 0: yield (); return
    for p in range(min(m, maxp), 0, -1):
        for rest in partitions(m - p, p): yield (p,) + rest
parts = list(partitions(n))
print("number of partitions of 10:", len(parts))
def logcount(part):
    # number of set partitions of {1..n} with these block sizes times prod (n_i-1)!  = n!/prod(n_i) / prod(mult!)
    from collections import Counter
    c = Counter(part)
    v = lgamma(n + 1) - sum(log(x) for x in part) - sum(lgamma(m + 1) for m in c.values())
    return v
def lp(part, th):  # probability of the allelic partition (unlabelled) under ESF
    k = len(part)
    return np.exp(logcount(part) + k * np.log(th) - (gammaln(th + n) - gammaln(th)))
th_test = 2.3
print("sum over partitions", sum(lp(p, th_test) for p in parts))
prior = lambda t: 1 / (np.log(2) * (t + 1) * (t + 2))
def post_quant(like, levels=(0.05, 0.5, 0.95), hi=np.inf):
    f = lambda t: prior(t) * like(t)
    # integrate in log scale
    g = lambda x: f(np.exp(x)) * np.exp(x)
    Z = quad(g, -20, 12, limit=400, points=[0, 1, 2])[0]
    out = []
    for a in levels:
        out.append(np.exp(brentq(lambda x: quad(g, -20, x, limit=400)[0] / Z - a, -20, 12, xtol=1e-10)))
    return np.array(out), Z
# exact posterior given full partition
qfull, Zfull = post_quant(lambda t: lp(obs, t))
# given K = 4
PK = lambda k, t: sum(lp(p, t) for p in parts if len(p) == k)
qK, ZK = post_quant(lambda t: PK(4, t))
print("posterior | partition", np.round(qfull, 4), " | K=4", np.round(qK, 4))
# ABC targets with K summary at eps
for eps in (0, 1, 2, 3):
    like = lambda t: sum(PK(k, t) for k in range(1, n + 1) if abs(k - 4) <= eps)
    q, Z = post_quant(like)
    print("ABC K eps", eps, "acceptance", round(Z, 4), "q05 med q95", np.round(q, 3), "relw", round((q[2]-q[0])/q[1], 3))
# ABC with largest block as summary, eps = 0 (exact posterior given max block = 5)
like = lambda t: sum(lp(p, t) for p in parts if p[0] == 5)
q, Z = post_quant(like); print("max block = 5: acceptance", round(Z, 4), np.round(q, 3), round((q[2]-q[0])/q[1], 3))
# ABC with number of singletons as summary, observed 1
like = lambda t: sum(lp(p, t) for p in parts if p.count(1) == 1)
q, Z = post_quant(like); print("singletons = 1: acceptance", round(Z, 4), np.round(q, 3), round((q[2]-q[0])/q[1], 3))
# Monte Carlo rejection ABC with 1e5 prior draws, Feller coupling
rng = np.random.default_rng(10)
M = 100000
w = 2 ** rng.random(M)            # inverse transform: F(theta) = log2(2(theta+1)/(theta+2))
thp = 2 * (w - 1) / (2 - w)
i = np.arange(1, n + 1)
B = rng.random((M, n)) < thp[:, None] / (thp[:, None] + i[None, :] - 1)
K = B.sum(1)
for eps in (0, 1, 2, 3):
    acc = np.abs(K - 4) <= eps
    q = np.quantile(thp[acc], [0.05, 0.5, 0.95])
    print("MC ABC eps", eps, "accepted", acc.sum(), "rate", acc.mean(), "q", np.round(q, 3))
print("prior median", 2*(np.sqrt(2)-1)/(2-np.sqrt(2)))
# MLE of theta from K=4, n=10
s = lambda t: np.sum(t / (t + np.arange(n))) - 4
print("MLE", brentq(s, 1e-6, 100))
