"""Pitman Yor, n=4, K=2, sigma=1/2, half horseshoe PH(0,1,2,1) prior on theta.
Likelihood in theta: (theta+sigma)/[theta+1]^3. Posterior kernel (t+1/2)/((t+1)^2(t+2)^2(t+3))."""
import sympy as sp
import mpmath as mp
t = sp.symbols('t', positive=True)
sig = sp.Rational(1, 2)
den = (t + 1) ** 2 * (t + 2) ** 2 * (t + 3)
def I(expr):
    f = sp.apart(sp.together(expr), t)
    val = sp.integrate(f, (t, 0, sp.oo))
    return sp.nsimplify(sp.simplify(val))
C = I((t + sig) / den)
M1 = I(t * (t + sig) / den)
P = I((t + sig) * (t + 2 * sig) / (den * (t + 4)))
print("C", C, float(C))
print("mean", float(M1 / C))
print("P(new type at draw 5) = E[(theta+2 sigma)/(theta+4)]", float(P / C))
print("partial fractions", sp.apart((t + sig) / den, t))
f = sp.lambdify(t, (t + sig) / den, 'mpmath')
Cq = mp.quad(f, [0, mp.inf])
print("quad", Cq)
cdf = lambda A: mp.quad(f, [0, A]) / Cq
lo, hi = mp.mpf(0), mp.mpf(100)
for _ in range(80):
    mid = (lo + hi) / 2
    if cdf(mid) < 0.5: lo = mid
    else: hi = mid
print("median", lo)
# Ewens quantiles by bisection
g = lambda x: x / ((x + 1) ** 2 * (x + 2) ** 2 * (x + 3))
Cg = mp.quad(g, [0, mp.inf])
for q in [0.025, 0.5, 0.975]:
    lo, hi = mp.mpf(0), mp.mpf(1000)
    for _ in range(80):
        mid = (lo + hi) / 2
        if mp.quad(g, [0, mid]) / Cg < q: lo = mid
        else: hi = mid
    print("Ewens n=4 K=2 quantile", q, mp.nstr(lo, 6))
