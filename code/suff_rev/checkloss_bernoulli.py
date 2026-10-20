"""Check loss worked example: theta ~ U(0,1) allele frequency, one gene sampled, Y | theta ~ Bern(theta).
Posterior Beta(2,1) (y=1, quantile sqrt u) or Beta(1,2) (y=0, quantile 1 - sqrt(1-u))."""
import sympy as sp, numpy as np
u, t, c = sp.symbols('u t c', positive=True)
rho = lambda z, uu: z * (uu - sp.Heaviside(-z))
# expected check loss for theta with density f on (0,1), at level u, constant c in (0,1)
def exp_loss(f, uu, cc):
    return sp.integrate((t - cc) * uu * f, (t, cc, 1)) + sp.integrate((t - cc) * (uu - 1) * f, (t, 0, cc))
f1 = 2 * t
f0 = 2 * (1 - t)
q1 = sp.sqrt(u); q0 = 1 - sp.sqrt(1 - u)
R1 = sp.integrate(sp.simplify(exp_loss(f1, u, q1)), (u, 0, 1))
R0 = sp.integrate(sp.simplify(exp_loss(f0, u, q0)), (u, 0, 1))
Rstar = sp.nsimplify(sp.simplify((R1 + R0) / 2))
print("Bayes risk R* =", Rstar, float(Rstar))
# risk of the map that ignores the data, H(u) = u (prior quantile)
Rprior = sp.integrate(sp.simplify(exp_loss(sp.Integer(1), u, u)), (u, 0, 1))
print("risk ignoring data =", sp.nsimplify(Rprior), float(Rprior))
print("excess =", sp.nsimplify(Rprior - Rstar), float(Rprior - Rstar))
W2sq = sp.integrate((sp.sqrt(u) - u) ** 2, (u, 0, 1))
print("W2^2 per y =", W2sq, "upper bound M/2 W2^2 with M=2:", W2sq)
# general identity: E rho_u(theta - c) = E[theta(u - 1(theta<c))] ... check at u = 1/2 for posterior median
print("posterior medians", float(q1.subs(u, sp.Rational(1, 2))), float(q0.subs(u, sp.Rational(1, 2))))
# prior risk of single-level loss: E rho_u(theta - u) for theta uniform = u(1-u)/2
print("E rho_u(U - u) =", sp.factor(sp.simplify(exp_loss(sp.Integer(1), u, u))))
print("E rho_u(theta - sqrt u | y=1) =", sp.factor(sp.simplify(exp_loss(f1, u, q1))))
# empirical quantile regression on simulated pairs
rng = np.random.default_rng(7)
for M in (200, 2000, 20000):
    th = rng.random(M); y = rng.random(M) < th
    lev = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    # minimise empirical check loss over c for each (u, y): the empirical u-quantile (lower)
    e1 = np.quantile(th[y], lev, method="inverted_cdf")
    e0 = np.quantile(th[~y], lev, method="inverted_cdf")
    print(M, "y=1 fitted", np.round(e1, 3), "exact", np.round(np.sqrt(lev), 3))
    print(M, "y=0 fitted", np.round(e0, 3), "exact", np.round(1 - np.sqrt(1 - lev), 3))
# check that empirical check loss minimiser = empirical quantile by brute force at u=0.25, M=2000 last sample
grid = np.linspace(0, 1, 100001)
uu = 0.25; z = th[y][None, :] - grid[:, None] if False else None
x = np.sort(th[y]); 
L = [np.mean((x - g) * (uu - (x < g))) for g in np.linspace(0, 1, 2001)]
print("brute force argmin", np.linspace(0, 1, 2001)[int(np.argmin(L))], "quantile", np.quantile(x, uu, method="inverted_cdf"))
