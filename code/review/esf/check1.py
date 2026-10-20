"""Review checks for chapter esf: small cases, butterfly fits, information tables."""
import numpy as np
from fractions import Fraction as Fr
from math import factorial, comb, log
from itertools import product
from scipy.special import digamma, polygamma, gammaln
from scipy.optimize import brentq, minimize
import sympy as sp

# Ten balls example
def tv_colour(n0, n1, k):
    n = n0 + n1
    from itertools import product as pr
    tot = 0
    for seq in pr([0, 1], repeat=k):
        c1 = sum(seq); c0 = k - c1
        w = Fr(n0, n) ** c0 * Fr(n1, n) ** c1
        # without
        p = Fr(1); a0, a1, m = n0, n1, n
        for s in seq:
            if s == 0:
                p *= Fr(a0, m); a0 -= 1
            else:
                p *= Fr(a1, m); a1 -= 1
            m -= 1
        tot += abs(w - p)
    return tot / 2
for k in (2, 3, 4):
    print("tenballs k", k, float(tv_colour(6, 4, k)))
print("ex1", tv_colour(3, 3, 2))
def pnk(n, k):
    p = 1.0
    for i in range(k): p *= 1 - i / n
    return 1 - p
print("p1000,10", pnk(1000, 10), "p1000,30", pnk(1000, 30), 30*29/2000)

# partitions
def partitions(n, m=None):
    if m is None: m = n
    if n == 0: yield []; return
    for f in range(min(n, m), 0, -1):
        for r in partitions(n - f, f): yield [f] + r
def rising(x, m):
    r = 1
    for i in range(m): r *= (x + i)
    return r
def esf_shape(sh, th):
    n = sum(sh); a = {}
    for s in sh: a[s] = a.get(s, 0) + 1
    c = Fr(factorial(n))
    for j, aj in a.items(): c /= (j ** aj) * factorial(aj)
    return c * th ** len(sh) / rising(th, n)
for n in (4, 5):
    for sh in partitions(n):
        print(n, sh, [str(esf_shape(sh, t)) for t in (Fr(1, 2), Fr(1), Fr(2))])
for n, th in ((4, Fr(2)), (5, Fr(1)), (5, Fr(2))):
    E = sum(len(sh) * esf_shape(sh, th) for sh in partitions(n))
    E2 = sum(len(sh) ** 2 * esf_shape(sh, th) for sh in partitions(n))
    print("n", n, "th", th, "EK", E, "E2", E2, "Var", E2 - E * E)
    EA = [sum(sh.count(j) * esf_shape(sh, th) for sh in partitions(n)) for j in range(1, n + 1)]
    print("  EA", [str(x) for x in EA])
print("I4(2)", Fr(641, 900) / 4, float(Fr(641, 900) / 4), "se", float((Fr(641, 900) / 4) ** -1) ** .5)

# PY shape
def py_shape(sh, al, th):
    n = sum(sh); a = {}
    for s in sh: a[s] = a.get(s, 0) + 1
    N = Fr(factorial(n))
    for j, aj in a.items(): N /= factorial(j) ** aj * factorial(aj)
    k = len(sh)
    num = Fr(1)
    for i in range(1, k): num *= th + i * al
    for s in sh: num *= rising(1 - al, s - 1)
    return N * num / rising(th + 1, n - 1)
for pars in ((Fr(1, 2), Fr(1)), (Fr(0), Fr(1)), (Fr(-1), Fr(2))):
    pr_ = [py_shape(sh, *pars) for sh in partitions(4)]
    print("py4", pars, [str(x) for x in pr_], "EK", sum(len(sh) * py_shape(sh, *pars) for sh in partitions(4)), sum(pr_))
al, th = sp.symbols('al th')
s3 = sp.simplify(((1-al)*(2-al) + 3*(1-al)*(th+al) + (th+al)*(th+2*al))/((th+1)*(th+2)))
print("ex9 sum", s3)

# Table esf n=8 theta=2
sh8 = sorted(partitions(8), key=lambda s: -esf_shape(s, Fr(2)))
for s in sh8[:13]: print(s, round(float(esf_shape(s, Fr(2))), 4))
print("count n8", len(sh8))

# Butterflies
a = np.array([118, 74, 44, 24, 29, 22, 20, 19, 20, 15, 12, 14, 6, 12, 6, 9, 9, 6, 10, 10, 11, 5, 3, 3])
j = np.arange(1, 25); n = int((a * j).sum()); K = int(a.sum()); print("n K", n, K)
EK = lambda t, n=n: t * (digamma(t + n) - digamma(t))
th = brentq(lambda u: EK(u) - K, 1, 2000)
I = lambda t: EK(t) / t**2 - (polygamma(1, t) - polygamma(1, t + n))
print("mle", th, "se", I(th) ** -.5, "VarK", th**2 * I(th))
def EAj(t, jj, n=n):
    return np.exp(np.log(t) - np.log(jj) + gammaln(n + 1) - gammaln(n - jj + 1) + gammaln(t + n - jj) - gammaln(t + n))
ea = EAj(th, j); ls = th * (n / (n + th)) ** j / j
print("Ewens EA", np.round(ea, 2)); print("logser", np.round(ls, 2)); print("maxdiff", np.max(np.abs(ea - ls)), "sum EA24", ea.sum())
print("pearson ewens", ((a - ea) ** 2 / ea).sum())
def pyEA(al, t, jj, n=n):
    return np.exp(gammaln(n + 1) - gammaln(jj + 1) - gammaln(n - jj + 1) + np.log(t + al) + gammaln(1 - al + jj - 1) - gammaln(1 - al)
                  + gammaln(t + al + 1 + n - jj - 1) - gammaln(t + al + 1) - (gammaln(t + 1 + n - 1) - gammaln(t + 1)))
def pyll(al, t):
    k = K
    # log P(A=a) including n! and combinatorial factors
    i = np.arange(1, k)
    ll = gammaln(n + 1) + np.sum(np.log(t + i * al)) - (gammaln(t + n) - gammaln(t + 1))
    ll += np.sum(-gammaln(a + 1) + a * (gammaln(j - al) - gammaln(1 - al) - gammaln(j + 1)))
    return ll
print("ll ewens", pyll(0.0, th))
res = minimize(lambda p: -pyll(p[0], np.exp(p[1])), [-0.4, np.log(300)], method='Nelder-Mead', options=dict(xatol=1e-8, fatol=1e-10, maxiter=5000))
alh, thh = res.x[0], np.exp(res.x[1]); print("py mle", alh, thh, -res.fun, "m", thh / -alh)
pe = pyEA(alh, thh, j); print("py EA", np.round(pe, 2)); print("pearson py", ((a - pe) ** 2 / pe).sum(), "sum", pe.sum(), "check sum all j", (np.arange(1, n+1) * pyEA(alh, thh, np.arange(1, n+1))).sum())
# new species
print("ewens new", EK(th, 2 * n) - K, "direct", EAj(th, np.arange(1, n+1)).sum())
def pyEK(al, t, n):
    return (t / al) * (np.exp(gammaln(t + al + n) - gammaln(t + n) + gammaln(t + 1) - gammaln(t + al + 1)) - 1)
print("py EK n", pyEK(alh, thh, n), "py new", pyEK(alh, thh, 2 * n) - pyEK(alh, thh, n), pyEK(alh, thh, 2*n) - K)
m = 735; aa = 0.4936
print("ex10", m * (1 - np.exp(gammaln((m - 1) * aa + n) - gammaln((m - 1) * aa) - gammaln(m * aa + n) + gammaln(m * aa))))
gt = sum((-1) ** (jj + 1) * aj for jj, aj in zip(j, a)); print("GT", gt)
F = (a * j * (j - 1)).sum() / (n * (n - 1)); print("F", F, 1 / F - 1)
print("GT expectation all j", sum((-1)**(jj+1)*EAj(th, jj) for jj in range(1, n+1)), "j<=24", sum((-1)**(jj+1)*EAj(th, jj) for jj in range(1, 25)))
print("ex8", sum(th / (th + i) for i in range(n, 2 * n)), th * log(2))

# Table kn
for t in (0.5, 2, 10):
    for nn in (100, 1000, 10000):
        e = t * (digamma(t + nn) - digamma(t)); ii = np.arange(nn); In = np.sum(ii / (t * (t + ii) ** 2))
        print("kn", t, nn, round(e, 3), round(In ** -.5, 3), "pred EK/logn", round(e / np.log(nn), 3))
# info gap table
for t in (2, 10, 0.5):
    for nn in (4, 10, 100, 1000, 10**4, 10**6):
        ii = np.arange(1, nn); In = np.sum(ii / (t * (t + ii) ** 2)); L = np.sum(1 / ii) / t
        print("info", t, nn, round(In, 3), round(L, 3), round(In / L, 2), "gap", round(L - In, 3))
