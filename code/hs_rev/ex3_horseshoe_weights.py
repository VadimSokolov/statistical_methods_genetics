# Horseshoe shrinkage weights for a few loci in standardized units z ~ N(theta, 1), theta | k ~ N(0, (1-k)/k)
# kappa = 1/(1+tau^2 lambda^2), lambda ~ C+(0,1). Moments by quadrature in kappa.
import numpy as np
from scipy import integrate
def post(z, tau):
    # density of kappa given tau: lambda ~ C+, kappa = 1/(1+tau^2 lam^2)
    # p(kappa) ∝ kappa^{-1/2}(1-kappa)^{-1/2} / (1 + (tau^2-1) kappa)   (Carvalho Polson Scott 2010)
    # marginal likelihood given kappa: z ~ N(0, 1/kappa) -> sqrt(kappa) exp(-kappa z^2/2)
    f = lambda k: k**-0.5*(1-k)**-0.5/(1+(tau**2-1)*k)*np.sqrt(k)*np.exp(-k*z*z/2)
    Z0 = integrate.quad(f, 0, 1, limit=200)[0]
    m = lambda g: integrate.quad(lambda k: g(k)*f(k), 0, 1, limit=200)[0]/Z0
    Ek = m(lambda k: k); Ek2 = m(lambda k: k*k)
    E1k = 1-Ek; E1k2 = 1-2*Ek+Ek2
    mean = E1k*z
    Et2 = z*z*E1k2 + E1k            # E[theta^2|z]
    var = Et2-mean**2
    return Ek, mean, var, Et2
zs = [0.5, 1.0, 2.0, 3.0, 5.0]
for tau in [1.0, 0.1]:
    print("tau", tau)
    tot = np.zeros(4)
    for z in zs:
        Ek, m, v, Et2 = post(z, tau)
        print(f"z={z:4.1f} Ek={Ek:.3f} m={m:.3f} s2={v:.3f} m2={m*m:.3f} Et2={Et2:.3f} z2-1={z*z-1:.2f} kopt={1/(1+0)}")
        tot += [z*z, m*m, Et2, z*z-1]
    print("sums z2, m2, Et2, z2-1:", np.round(tot, 3))
# Monte Carlo check of tau=1 moments via lambda sampling
rng = np.random.default_rng(1)
lam = np.abs(rng.standard_cauchy(4_000_000))
for z in [1.0, 3.0]:
    k = 1/(1+lam**2); w = np.sqrt(k)*np.exp(-k*z*z/2)
    print("MC z", z, "Ek", np.sum(w*k)/np.sum(w), "Et2", np.sum(w*(z*z*(1-k)**2+(1-k)))/np.sum(w))
