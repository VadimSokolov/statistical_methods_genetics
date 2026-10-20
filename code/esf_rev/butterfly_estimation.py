"""Estimation of theta for Corbet's butterflies (table truncated at j = 24):
MLE by Newton, standard errors, K/log n, homozygosity estimator, likelihood interval,
and a simulation of the sampling distributions at theta = 164."""
import numpy as np
from scipy.special import digamma, polygamma, gammaln
from scipy.optimize import brentq

a = np.array([118, 74, 44, 24, 29, 22, 20, 19, 20, 15, 12, 14,
              6, 12, 6, 9, 9, 6, 10, 10, 11, 5, 3, 3])
j = np.arange(1, 25)
n = int((a * j).sum()); K = int(a.sum())
print("n", n, "K", K)

EK = lambda t: t * (digamma(t + n) - digamma(t))
VK = lambda t: EK(t) - t**2 * (polygamma(1, t) - polygamma(1, t + n))
I = lambda t: EK(t) / t**2 - (polygamma(1, t) - polygamma(1, t + n))
loglik = lambda t: K * np.log(t) + gammaln(t) - gammaln(t + n)

# Newton on score: s(t) = K/t - (psi(t+n)-psi(t)); observed info = K/t^2 - (psi'(t)-psi'(t+n))
t = K / np.log(n)
print("start K/log n", t)
for it in range(6):
    s = K / t - (digamma(t + n) - digamma(t))
    J = K / t**2 - (polygamma(1, t) - polygamma(1, t + n))
    t = t + s / J
    print(" iter", it + 1, t, "score", s)
th = brentq(lambda u: EK(u) - K, 1, 2000)
print("MLE", th, "I", I(th), "se", I(th) ** -0.5, "VK at mle", VK(th), "dEK/dtheta", VK(th) / th)
print("asymptotic se sqrt(theta/log n)", np.sqrt(th / np.log(n)))
# likelihood interval (drop 1.92)
lm = loglik(th)
lo = brentq(lambda u: loglik(u) - lm + 1.92, 50, th)
hi = brentq(lambda u: loglik(u) - lm + 1.92, th, 1000)
print("LR 95% interval", lo, hi, "Wald", th - 1.96 * I(th) ** -0.5, th + 1.96 * I(th) ** -0.5)
# delta method on log scale interval
se_log = I(th) ** -0.5 / th
print("log scale interval", th * np.exp(-1.96 * se_log), th * np.exp(1.96 * se_log))

# homozygosity estimator
F = (a * j * (j - 1)).sum() / (n * (n - 1))
print("F hat", F, "theta_F", 1 / F - 1)
print("expected homozygosity at mle", 1 / (1 + th))

# simulation of K and F at theta = 164, n = 3306 through the Hoppe urn
rng = np.random.default_rng(20261017)
R = 2000
thv = 164.0
Ks = np.empty(R); Fs = np.empty(R); mle = np.empty(R)
# full urn needed for F: use index labels
def urn(thv, n, rng):
    lab = np.empty(n, dtype=np.int64)
    k = 0
    u = rng.random(n); w = rng.random(n)
    for i in range(n):
        if u[i] < thv / (thv + i):
            lab[i] = k; k += 1
        else:
            lab[i] = lab[int(w[i] * i)]
    c = np.bincount(lab)
    return k, (c * (c - 1)).sum() / (n * (n - 1))
for r in range(R):
    k, f = urn(thv, n, rng)
    Ks[r] = k; Fs[r] = f
    mle[r] = brentq(lambda u: EK(u) - k, 1, 5000)
thF = 1 / Fs - 1
print("sim mean K", Ks.mean(), "EK", EK(thv), "sd K", Ks.std(), "sqrt VK", np.sqrt(VK(thv)))
print("sim mle mean", mle.mean(), "sd", mle.std(), "I^-1/2", I(thv) ** -0.5)
print("sim theta_F mean", thF.mean(), "median", np.median(thF), "sd", thF.std())
print("sim F mean", Fs.mean(), "1/(1+th)", 1 / (1 + thv))
print("K/log n mean", (Ks / np.log(n)).mean(), "pred", EK(thv) / np.log(n))

# homozygosity under the two parameter (finite pool) fit: Dirichlet multinomial, m = 735, a = 0.4936
m, aa = 735, 0.4936
print("rho two parameter", (1 + aa) / (1 + m * aa))
Fd = np.empty(R)
for r in range(R):
    p = rng.dirichlet(np.full(m, aa))
    c = rng.multinomial(n, p)
    Fd[r] = (c * (c - 1)).sum() / (n * (n - 1))
print("two param F mean", Fd.mean(), "sd", Fd.std(), "q05 q95", np.quantile(Fd, [0.05, 0.95]))
print("Ewens F q05 q95", np.quantile(Fs, [0.005, 0.05, 0.95]), "min", Fs.min())

# fair comparison: homozygosity contribution of species with at most 24 individuals
def trunc_stats(c):
    c = c[(c > 0) & (c <= 24)]
    return (c * (c - 1)).sum() / (n * (n - 1)), len(c), c.sum()
def urn_counts(thv, n, rng):
    lab = np.empty(n, dtype=np.int64); k = 0
    u = rng.random(n); w = rng.random(n)
    for i in range(n):
        if u[i] < thv / (thv + i):
            lab[i] = k; k += 1
        else:
            lab[i] = lab[int(w[i] * i)]
    return np.bincount(lab)
R2 = 1000
G1 = np.array([trunc_stats(urn_counts(thv, n, rng)) for _ in range(R2)])
G2 = np.array([trunc_stats(rng.multinomial(n, rng.dirichlet(np.full(m, aa)))) for _ in range(R2)])
for name, G in (("Ewens", G1), ("two param", G2)):
    print(name, "truncated F mean sd", G[:, 0].mean(), G[:, 0].std(), "P(<= obs)", (G[:, 0] <= F).mean(),
          "species<=24", G[:, 1].mean(), "individuals<=24", G[:, 2].mean())
