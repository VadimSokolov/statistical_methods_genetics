"""S is not sufficient under the infinite sites model: exact calculation for n = 3,
and the general n identity E[l_{n-1} L]/E[l_{n-1}] = 2 + 2 a_n versus E[L^2]/E[L] = 2 a_n + 2 b_n / a_n."""
import sympy as sp
import numpy as np
th, t3, t2 = sp.symbols('theta t3 t2', positive=True)
lam = th / 2
# n = 3: T3 ~ Exp(3), T2 ~ Exp(1); l1 = 3 T3 + T2, l2 = T2
l1, l2 = 3 * t3 + t2, t2
dens = 3 * sp.exp(-3 * t3) * sp.exp(-t2)
def p(a, b):
    f = sp.exp(-lam * (l1 + l2)) * (lam * l1) ** a * (lam * l2) ** b / (sp.factorial(a) * sp.factorial(b)) * dens
    return sp.simplify(sp.integrate(sp.integrate(sp.expand(f), (t3, 0, sp.oo)), (t2, 0, sp.oo)))
P = {(a, b): sp.factor(p(a, b)) for a in range(4) for b in range(4) if a + b <= 3}
for k, v in P.items(): print(k, v)
# check P(S=s) against geometric sum
for s in range(4):
    tot = sp.factor(sum(v for (a, b), v in P.items() if a + b == s))
    # geometric convolution for n=3: j=1,2
    g = sum((th / (th + 1)) ** i * (1 / (th + 1)) * (th / (th + 2)) ** (s - i) * (2 / (th + 2)) for i in range(s + 1))
    print("S", s, sp.simplify(tot - g) == 0, tot)
print("P(xi=(0,1) | S=1) =", sp.factor(P[(0, 1)] / (P[(0, 1)] + P[(1, 0)])))
r1 = sp.factor(P[(0, 1)] / P[(1, 0)]); print("LR (0,1)/(1,0):", r1)
r2 = sp.factor(P[(0, 2)] / P[(2, 0)]); print("LR (0,2)/(2,0):", r2)
r3 = sp.factor(P[(1, 1)] / P[(2, 0)]); print("LR (1,1)/(2,0):", r3)
for v in (0.5, 1, 2, 4, 8, 1000):
    print(v, [float(x.subs(th, v)) for x in (r1, r2, r3)])
# likelihood ratio theta=1 vs theta=4 for each S=2 data set
for k in [(2, 0), (1, 1), (0, 2)]:
    print(k, "L(1)/L(4) =", float(P[k].subs(th, 1) / P[k].subs(th, 4)), sp.nsimplify(P[k].subs(th, 1) / P[k].subs(th, 4)))
# posterior medians under log uniform prior on [0.5, 30]
from scipy.integrate import quad
from scipy.optimize import brentq
for k in [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]:
    f = sp.lambdify(th, P[k] / th)
    Z = quad(f, 0.5, 30)[0]
    q = [brentq(lambda x: quad(f, 0.5, x)[0] / Z - a, 0.5, 30) for a in (0.05, 0.5, 0.95)]
    print(k, "posterior q05, median, q95", np.round(q, 3))
f = sp.lambdify(th, (P[(1, 0)] + P[(0, 1)]) / th); Z = quad(f, 0.5, 30)[0]
print("given S=1 only median", brentq(lambda x: quad(f, 0.5, x)[0] / Z - 0.5, 0.5, 30))
f = sp.lambdify(th, (P[(2, 0)] + P[(1, 1)] + P[(0, 2)]) / th); Z = quad(f, 0.5, 30)[0]
print("given S=2 only median", brentq(lambda x: quad(f, 0.5, x)[0] / Z - 0.5, 0.5, 30))

# general n check of the two ratios
for n in (3, 5, 10, 30):
    a = sum(1 / j for j in range(1, n)); b = sum(1 / j**2 for j in range(1, n))
    print(n, "2+2a", 2 + 2 * a, "2a+2b/a", 2 * a + 2 * b / a, "P(e_{n-1}|S=1,theta->0)", 1 / ((n - 1) * a))
# simulation check n = 10 of P(xi = e_{n-1} | S = 1, theta) at two theta
rng = np.random.default_rng(3)
def sim(theta, n, R):
    sizes = np.ones((R, n), int); xi = np.zeros((R, n + 1), int); rows = np.arange(R)
    for k in range(n, 1, -1):
        T = rng.exponential(1 / (k * (k - 1) / 2), R)
        mut = rng.poisson((theta * T / 2)[:, None] * np.ones((1, k)))
        np.add.at(xi, (np.repeat(rows, k), sizes[:, :k].ravel()), mut.ravel())
        i = rng.integers(0, k, R); j = rng.integers(0, k - 1, R); j = j + (j >= i)
        sizes[rows, i] += sizes[rows, j]; sizes[rows, j] = sizes[rows, k - 1]
    return xi[:, 1:n]
for thv in (0.2, 3.0):
    x = sim(thv, 3, 400000); S = x.sum(1)
    print("n=3 sim theta", thv, "P(doubleton|S=1)", (x[S == 1, 1] == 1).mean(), "exact", (2 + thv) / (6 + 4 * thv))
