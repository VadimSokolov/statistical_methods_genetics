"""Reference posterior quantiles for the growth model table of Chapter coal (the table of reference quantiles).

Model: n = 30 sequences, theta ~ Ga(4, scale 25), growth rate g ~ Uniform(0, 20), and S given the tree
length L is Poisson(theta L / 2). The marginal over theta is analytic, so the posterior of theta given S
is a gamma mixture over simulated tree lengths and the posterior of g is a weighted histogram over the
simulated growth rates. The tree lengths come from scrambled Sobol points in eight independent groups.

Usage: python3 -I code/coal_rev/ref_posterior_qmc.py M OUT.json SEED
M is the base 2 logarithm of the number of Sobol points per group (at least 19). The rows printed in the
chapter come from `ref_posterior_qmc.py 25 /tmp/coal_ref.json 9090`, which is 8 x 2^25 = 2.7e8 simulated
tree lengths and takes several minutes. OUT.json receives the 5, 25, 50, 75 and 95 percent posterior
quantiles of theta and of g for S = 40, 100, 200, with standard errors across the eight groups.
"""
# Independent reference posterior for the growth model of coal.tex Section reference.
# Model: n=30, theta ~ Ga(4, scale 25), g ~ Unif(0,20); coalescent at constant rate in
# rescaled time Lambda, real time t = log(1+g Lambda)/g; S | L ~ Poisson(theta L/2).
# Marginal over theta is analytic: theta | s, L ~ Ga(s+4, rate 1/25 + L/2), weight
# (L/2)^s / (1/25+L/2)^(s+4). So no theta grid is needed; g uses fine bins (width 0.005)
# whose CDF is exact at the bin edges. MC over (g, L) from the prior, K independent groups.
import numpy as np, sys, json
from scipy import special, optimize
n = 30
ks = np.arange(n, 1, -1).astype(float)
rates = ks*(ks-1)/2
S_list = [40, 100, 200]
rate0 = 1/25.0
P = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
NB = 400000
ledges = np.exp(np.linspace(np.log(1e-4), np.log(80.0), NB+1))
lcent = np.sqrt(ledges[:-1]*ledges[1:])
GB = 4000
gedges = np.linspace(0, 20, GB+1)
def lwmax(s):
    x = s/100.0
    return s*np.log(x) - (s+4)*np.log(rate0 + x)
K = 8
from scipy.stats import qmc
m = int(sys.argv[1]); chunk = 2**19
HL = np.zeros((K, 3, NB)); HG = np.zeros((K, 3, GB))
sumL = 0.0; cnt = 0
rng = np.random.default_rng(int(sys.argv[3]))
for grp in range(K):
    eng = qmc.Sobol(d=n, scramble=True, seed=rng)
    for b in range(2**m//chunk):
        u = eng.random(chunk)
        g = np.maximum(20*u[:, 0], 1e-12)
        E = -np.log1p(-u[:, 1:])          # columns k = 2, 3, ..., n (most important first)
        T = E[:, ::-1]/rates               # reorder to time order k = n..2
        Lam = np.cumsum(T, axis=1)
        tau = np.log1p(g[:, None]*Lam)/g[:, None]
        dur = np.diff(tau, axis=1, prepend=0.0)
        L = dur @ ks
        sumL += L.sum(); cnt += chunk
        lb = np.searchsorted(ledges, L) - 1
        assert lb.min() >= 0 and lb.max() < NB
        gb = np.minimum((g/20*GB).astype(int), GB-1)
        for i, s in enumerate(S_list):
            w = np.exp(s*np.log(L/2) - (s+4)*np.log(rate0 + L/2) - lwmax(s))
            HL[grp, i] += np.bincount(lb, weights=w, minlength=NB)
            HG[grp, i] += np.bincount(gb, weights=w, minlength=GB)
# sanity: constant size mean tree length and prior mean of S under growth
T = rng.standard_exponential((200000, n-1))/rates
print("g=0 mean L", (T@ks).mean(), "theory", 2*sum(1/k for k in range(1, n)))
print("growth prior E[L]", sumL/cnt, "=> E[S] =", 100*sumL/cnt/2)
def thq(W, s):
    m = W > W.max()*1e-16
    Wn = W[m]/W[m].sum(); bb = rate0 + lcent[m]/2
    cdf = lambda t: np.dot(Wn, special.gammainc(s+4, bb*t))
    mean = np.dot(Wn, (s+4)/bb)
    return np.array([optimize.brentq(lambda t: cdf(t)-p, 1e-6, 2000, xtol=1e-7) for p in P]), mean
def gq(W):
    c = np.concatenate([[0.0], np.cumsum(W)]); c /= c[-1]
    return np.interp(P, c, gedges)
out = {}
for i, s in enumerate(S_list):
    tq, tm = thq(HL[:, i].sum(0), s); gqq = gq(HG[:, i].sum(0))
    tg = np.array([thq(HL[k, i], s)[0] for k in range(K)])
    gg = np.array([gq(HG[k, i]) for k in range(K)])
    tse = tg.std(0, ddof=1)/np.sqrt(K); gse = gg.std(0, ddof=1)/np.sqrt(K)
    print(f"S={s} theta q:", np.round(tq, 4), " SE", np.round(tse, 4), " mean", round(tm, 3))
    print(f"S={s} g     q:", np.round(gqq, 4), " SE", np.round(gse, 5))
    out[s] = dict(theta=tq.tolist(), theta_se=tse.tolist(), g=gqq.tolist(), g_se=gse.tolist())
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "/dev/null", "w"), indent=1)
