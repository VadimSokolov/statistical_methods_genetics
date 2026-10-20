"""Section 6: horseshoe shrinkage weight, scale mixtures, Beta augmentation, Polya Gamma."""
import numpy as np
from scipy import integrate, special, stats

# horseshoe: y ~ N(beta,1), beta ~ N(0, lambda^2), lambda ~ C+(0,1), tau=1
# p(kappa | y) propto (1-kappa)^{-1/2} exp(-kappa y^2/2)
def Ek(y):
    f = lambda k: (1-k)**-0.5*np.exp(-k*y*y/2)
    Z = integrate.quad(f, 0, 1)[0]
    return integrate.quad(lambda k: k*f(k), 0, 1)[0]/Z
for y in [0, 0.5, 1, 2, 3, 5, 10]:
    e = Ek(y)
    print("y", y, "E[kappa|y]", round(e,4), "post mean", round((1-e)*y,4))
# prior on kappa check: lambda ~ C+ => kappa = 1/(1+lambda^2) ~ Be(1/2,1/2)
rng = np.random.default_rng(5)
lam = np.abs(rng.standard_cauchy(10**6)); kap = 1/(1+lam**2)
print("kappa quantiles", np.quantile(kap,[.1,.25,.5,.75,.9]).round(4), "Beta(1/2,1/2)", stats.beta(.5,.5).ppf([.1,.25,.5,.75,.9]).round(4))
# with tau: kappa = 1/(1+tau^2 lambda^2); P(kappa < 1/2) = P(lambda > 1/tau) = 1 - (2/pi) atan(1/tau)
for tau in [1, 0.1, 0.01]:
    print("tau", tau, "P(kappa<1/2)", round(1-2/np.pi*np.arctan(1/tau),4))

# scale mixtures: Student t_nu = N(0, 1/w), w ~ Ga(nu/2, nu/2); Laplace(b) = N(0, v), v ~ Exp(rate 1/(2 b^2))
x = 1.3; nu = 3
mix = integrate.quad(lambda w: stats.norm.pdf(x, 0, 1/np.sqrt(w))*stats.gamma.pdf(w, nu/2, scale=2/nu), 0, np.inf)[0]
print("t3 density at 1.3", round(stats.t.pdf(x, nu),6), "mixture", round(mix,6))
b = 1.0
mixL = integrate.quad(lambda v: stats.norm.pdf(x, 0, np.sqrt(v))*stats.expon.pdf(v, scale=2*b*b), 0, np.inf)[0]
print("Laplace density", round(stats.laplace.pdf(x, scale=b),6), "mixture", round(mixL,6))
# horseshoe marginal prior density bounds (Carvalho Polson Scott): K/2 log(1+4/b^2) < p(b) < K log(1+2/b^2), K = 1/sqrt(2 pi^3)
K = 1/np.sqrt(2*np.pi**3)
for bb in [0.1, 1, 3]:
    p = integrate.quad(lambda l: stats.norm.pdf(bb,0,l)*2/(np.pi*(1+l*l)), 0, np.inf, limit=200)[0]
    print("hs prior density at", bb, round(p,5), "bounds", round(K/2*np.log(1+4/bb**2),5), round(K*np.log(1+2/bb**2),5))

# Beta augmentation Gamma(th)/Gamma(th+n) = 1/Gamma(n) int eta^{th-1}(1-eta)^{n-1}
th, n = 2.5, 10
lhs = np.exp(special.gammaln(th)-special.gammaln(th+n))
rhs = integrate.quad(lambda e: e**(th-1)*(1-e)**(n-1), 0, 1)[0]/special.gamma(n)
print("Beta aug", lhs, rhs)
# Gibbs sampler for Ewens theta: K=8, n=30, prior Ga(1,0.2)
Kobs, n = 8, 30
a0, b0 = 1.0, 0.2
th = 3.0; draws = []
for it in range(60000):
    eta = rng.beta(th, n)
    th = rng.gamma(Kobs+a0, 1/(b0-np.log(eta)))
    if it >= 2000: draws.append(th)
draws = np.array(draws)
grid = np.linspace(1e-4, 40, 400001)
logp = (Kobs+a0-1)*np.log(grid) - b0*grid + special.gammaln(grid) - special.gammaln(grid+n)
w = np.exp(logp-logp.max()); w /= w.sum()
print("Gibbs mean", draws.mean().round(3), "exact", (grid*w).sum().round(3), "Gibbs sd", draws.std().round(3), "exact", np.sqrt((grid**2*w).sum()-(grid*w).sum()**2).round(3))

# Polya Gamma: E exp(-t w) for PG(b,0) = cosh(sqrt(t/2))^{-b}; series w = (1/(2 pi^2)) sum g_k/(k-1/2)^2
bpg, t = 1.0, 0.7
kk = np.arange(1, 200001)
prod = np.exp(-bpg*np.log1p(t/(2*np.pi**2*(kk-0.5)**2)).sum())
print("PG Laplace", prod, np.cosh(np.sqrt(t/2))**-bpg)
# PG mean E[w | b, c] = b/(2c) tanh(c/2): Monte Carlo from truncated series
c = 1.5
g = rng.gamma(bpg, 1.0, size=(20000, 400))
k = np.arange(1, 401)
w = (g/((k-0.5)**2 + c*c/(4*np.pi**2))).sum(1)/(2*np.pi**2)
print("PG(1,1.5) mean MC", w.mean().round(4), "formula", round(bpg/(2*c)*np.tanh(c/2),4))
# logistic identity check: e^{a psi}/(1+e^psi)^b = 2^{-b} e^{kappa psi} E[exp(-w psi^2/2)], w~PG(b,0)
a, bb, psi = 1, 1, 0.8
lhs = np.exp(a*psi)/(1+np.exp(psi))**bb
rhs = 2**-bb*np.exp((a-bb/2)*psi)*np.cosh(np.sqrt(psi**2/4))**-bb
print("PG identity", lhs, rhs)
# PIG(0) mean = pi^2/12 check from series X0 = sum (4k^2)^{-1} / G_k, G_k ~ Ga(3/2,1)? print formula only
print("pi^2/12", np.pi**2/12)
k = np.arange(1, 2_000_001); c = 1.5
print("PG(1,1.5) mean by series", ((1/((k-0.5)**2 + c*c/(4*np.pi**2))).sum()/(2*np.pi**2)).round(5), "formula", round(1/(2*c)*np.tanh(c/2),5))
