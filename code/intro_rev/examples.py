"""Numerical checks for the worked examples in chapters/intro.tex."""
import math
from fractions import Fraction
import numpy as np
from scipy import stats, special

print("=== Mendel: round vs wrinkled")
r, w = 5474, 1850
n = r + w
er, ew = 0.75 * n, 0.25 * n
x2 = (r - er) ** 2 / er + (w - ew) ** 2 / ew
print("n", n, "ratio", r / w, "expected", er, ew, "chi2", x2, "p", stats.chi2.sf(x2, 1))
print("sd of round count", math.sqrt(n * .75 * .25), "z", (r - er) / math.sqrt(n * .75 * .25))
print("observed fraction", r / n)
# yellow vs green
y, g = 6022, 2001
n2 = y + g
x2b = (y - .75 * n2) ** 2 / (.75 * n2) + (g - .25 * n2) ** 2 / (.25 * n2)
print("yellow/green n", n2, "ratio", y / g, "chi2", x2b, "p", stats.chi2.sf(x2b, 1))
# dihybrid
obs = np.array([315, 101, 108, 32]); m = obs.sum()
exp = m * np.array([9, 3, 3, 1]) / 16
x2c = ((obs - exp) ** 2 / exp).sum()
print("dihybrid n", m, "exp", exp, "chi2", x2c, "p", stats.chi2.sf(x2c, 3))

print("=== Hardy Weinberg")
P = np.array([0.6, 0.2, 0.2])  # AA Aa aa
p = P[0] + P[1] / 2
print("p", p, "next gen", p * p, 2 * p * (1 - p), (1 - p) ** 2)
# random mating explicitly via gametes
gam = np.array([p, 1 - p])
print("explicit", np.outer(gam, gam))
q = math.sqrt(1 / 2500)
print("q", q, "carrier 2pq", 2 * q * (1 - q), "1 in", 1 / (2 * q * (1 - q)),
      "carriers per affected", 2 * q * (1 - q) / q ** 2)
# hypothetical sample of 500 genotypes test
cnt = np.array([180, 220, 100]); N = cnt.sum()
ph = (2 * cnt[0] + cnt[1]) / (2 * N)
e = N * np.array([ph ** 2, 2 * ph * (1 - ph), (1 - ph) ** 2])
x2h = ((cnt - e) ** 2 / e).sum()
Fhat = 1 - cnt[1] / e[1]
print("HW sample p", ph, "expected", e, "chi2", x2h, "p", stats.chi2.sf(x2h, 1), "F", Fhat, "N F^2", N * Fhat ** 2)

print("=== Fisher 1918 one locus")
a, d, pA = 1.0, 0.5, 0.3
qa = 1 - pA
freqs = np.array([pA ** 2, 2 * pA * qa, qa ** 2])
G = np.array([a, d, -a])
x = np.array([2, 1, 0])  # count of A
mu = freqs @ G
VG = freqs @ (G - mu) ** 2
xb = freqs @ x
beta = (freqs @ ((x - xb) * (G - mu))) / (freqs @ (x - xb) ** 2)
alpha = a + d * (qa - pA)
VA = 2 * pA * qa * alpha ** 2
VD = (2 * pA * qa * d) ** 2
fitted = mu + beta * (x - xb)
print("mean", mu, "VG", VG, "alpha", alpha, "beta(LS)", beta, "VA", VA, "VD", VD, "VA+VD", VA + VD)
print("breeding values", fitted, "dominance devs", G - fitted)
print("VA via LS", freqs @ (fitted - mu) ** 2, "VD via LS", freqs @ (G - fitted) ** 2)
VE = 1.0
VP = VG + VE
print("VP", VP, "h2 narrow", VA / VP, "H2 broad", VG / VP,
      "r parent offspring", VA / 2 / VP, "r full sibs", (VA / 2 + VD / 4) / VP)

print("=== Haldane 2s vs Kimura")
def branching(s):
    # survival prob of Poisson(1+s) Galton Watson: pi = 1 - exp(-(1+s) pi)
    from scipy.optimize import brentq
    if s <= 0:
        return 0.0
    return brentq(lambda z: z - 1 + math.exp(-(1 + s) * z), 1e-12, 1.0)

def kimura(s, N):
    p = 1 / (2 * N)
    if s == 0:
        return p
    return (-math.expm1(-4 * N * s * p)) / (-math.expm1(-4 * N * s))

def wf_exact(s, N):
    """Exact fixation prob for diploid WF with genic selection 1, 1+s, 1+2s (random union of gametes)."""
    M = 2 * N
    i = np.arange(M + 1)
    p = i / M
    wbar = 1 + 2 * s * p
    pp = p * (1 + s) / (1 + 2 * s * p) if s != 0 else p
    # careful: with genotype fitness 1,1+s,1+2s (A additive), p' = p(1+s+s p)/(1+2 s p)
    pp = p * (1 + s + s * p) / wbar
    T = stats.binom.pmf(i[None, :], M, pp[:, None])
    # absorbing at 0 and M; solve u = T u with u0=0, uM=1
    A = np.eye(M + 1) - T
    A[0, :] = 0; A[0, 0] = 1
    A[M, :] = 0; A[M, M] = 1
    b = np.zeros(M + 1); b[M] = 1
    u = np.linalg.solve(A, b)
    return u[1]

N = 500
for s in [0.05, 0.01, 0.001, 0.0001, 0.0]:
    print(f"N={N} s={s}: 2s={2*s:.6f} branching={branching(s):.6f} kimura={kimura(s,N):.6f} wf={wf_exact(s,N):.6f} 4Ns={4*N*s}")

print("=== Ewens formula n=5")
def partitions(n, maxp=None):
    if maxp is None:
        maxp = n
    if n == 0:
        yield []
        return
    for k in range(min(n, maxp), 0, -1):
        for rest in partitions(n - k, k):
            yield [k] + rest

def esf(part, th, n):
    a = {}
    for j in part:
        a[j] = a.get(j, 0) + 1
    rising = Fraction(1)
    for i in range(n):
        rising *= (th + i)
    val = Fraction(math.factorial(n)) / rising
    for j, aj in a.items():
        val *= (Fraction(th) / j) ** aj / math.factorial(aj)
    return val

n = 5
for th in [Fraction(1), Fraction(2)]:
    tot = 0
    print("theta", th)
    for part in partitions(n):
        v = esf(part, th, n)
        tot += v
        print("  ", part, v, float(v))
    print("   total", tot)
    EK = sum(th / (th + i) for i in range(n))
    print("   E K", EK, float(EK))
# conditional on K=2
for th in [Fraction(1), Fraction(2), Fraction(7, 3)]:
    v41 = esf([4, 1], th, n); v32 = esf([3, 2], th, n)
    print("cond K=2 theta", th, v41 / (v41 + v32), v32 / (v41 + v32))
# P(K=k) via Stirling numbers
for th in [1, 2]:
    s1 = [abs(special.stirling2(5, k)) for k in range(1, 6)]
print("unsigned stirling first kind n=5:", [24, 50, 35, 10, 1])
for th in [Fraction(1), Fraction(2)]:
    rising = 1
    for i in range(5):
        rising *= (th + i)
    print("P(K=k) theta", th, [Fraction(c) * th ** k / rising for c, k in zip([24, 50, 35, 10, 1], range(1, 6))])

print("=== coalescent n=5")
n = 5
ET = {k: Fraction(2, k * (k - 1)) for k in range(2, n + 1)}
for k in range(n, 1, -1):
    print(k, math.comb(k, 2), ET[k], float(ET[k]), "var", ET[k] ** 2)
Tmrca = sum(ET.values()); L = sum(k * ET[k] for k in ET)
VarT = sum(v ** 2 for v in ET.values())
an = sum(Fraction(1, j) for j in range(1, n))
print("E TMRCA", Tmrca, float(Tmrca), "sd", math.sqrt(VarT), "E L", L, float(L), "a_n", an, float(an))
print("share of T2", float(ET[2] / Tmrca))
Ne = 10000
print("generations (2Ne units)", float(Tmrca) * 2 * Ne, "T2", 2 * Ne)
rng = np.random.default_rng(1)
R = 200000
Ts = np.zeros(R); Ls = np.zeros(R)
for k in range(n, 1, -1):
    t = rng.exponential(1 / math.comb(k, 2), R)
    Ts += t; Ls += k * t
print("sim TMRCA", Ts.mean(), Ts.std(), "sim L", Ls.mean())
th = 5
print("E S with theta=5", th * float(an), "Var S", th * float(an) + th ** 2 * float(sum(Fraction(1, j * j) for j in range(1, n))))
