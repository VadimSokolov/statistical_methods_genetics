# NOTE (round 3): this script reproduces the UNCENSORED analysis (the 24 tabulated classes taken
# as the whole sample, n = 3306, K = 501, theta hat 163.99), which chapter rc now quotes only as the
# "ignoring the censoring" comparison (Section on Corbet's butterflies, Example rc:ex:butterflies).
# The censored analysis the chapter uses (n hat = 11068, K = 620) is in butterflies_censored.py
# and butterflies_censored_residue.py.
# Only its last line (the butterflies run) is uncensored; the sampler itself is still current.
"""Half horseshoe PH(0,1,2,1): exact inverse cdf draw, the latent exponential scale tau,
and an Escobar and West Gibbs sampler for the Ewens theta with every block exact.
Vectorised over independent chains."""
import numpy as np
rng = np.random.default_rng(20261017)

# 1. exact inverse cdf: F(A) = log{2(A+1)/(A+2)}/log 2, so A = 2(2^U - 1)/(2 - 2^U)
U = rng.random(10 ** 6)
A = 2 * (2 ** U - 1) / (2 - 2 ** U)
print("inverse cdf median", np.median(A), "exact median", 2 * (np.sqrt(2) - 1) / (2 - np.sqrt(2)))
print("P(A<=1) mc", np.mean(A <= 1), "exact", np.log(4 / 3) / np.log(2))
print("worked draws, U=0.1,0.5,0.9:", [2 * (2 ** u - 1) / (2 - 2 ** u) for u in (0.1, 0.5, 0.9)])

# 2. two stage draw through tau: alpha | tau ~ Exp(tau), tau ~ g(tau)/(tau log 2), g = e^-t - e^-2t
#    check by Gibbs on (alpha, tau): tau | alpha ~ Exp(1+alpha) + Exp(2+alpha)
M = 20000
al = np.ones(M)
for it in range(300):
    tau = rng.exponential(1 / (1 + al)) + rng.exponential(1 / (2 + al))
    al = rng.exponential(1 / tau)
print("Gibbs (alpha,tau) median", np.median(al), "P(alpha<=1)", np.mean(al <= 1))

# 3. Escobar and West with the half horseshoe for the Ewens theta given (n, k)
def ew_hh(n, k, chains=4000, iters=400, burn=100):
    th = np.full(chains, 1.0)
    keep = []
    for it in range(iters):
        tau = rng.exponential(1 / (1 + th)) + rng.exponential(1 / (2 + th))
        eta = rng.beta(th + 1, n)
        rate = tau - np.log(eta)
        odds = k / (n * rate)           # weight of Ga(k+1) against Ga(k)
        pick = rng.random(chains) < odds / (1 + odds)
        shape = np.where(pick, k + 1.0, float(k))
        th = rng.gamma(shape, 1 / rate)
        if it >= burn:
            keep.append(th.copy())
    s = np.concatenate(keep)
    return s

s = ew_hh(4, 2)
print("n=4,k=2: mean", s.mean(), "median", np.median(s), "q", np.quantile(s, [0.025, 0.975]),
      "P(new)", np.mean(s / (s + 4)), "draws", s.size)
s = ew_hh(3306, 501, chains=2000, iters=300, burn=100)
print("butterflies: mean", s.mean(), "median", np.median(s), "q", np.quantile(s, [0.025, 0.975]),
      "P(new)", np.mean(s / (s + 3306)))
