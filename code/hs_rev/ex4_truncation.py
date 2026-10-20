# Truncation selection on measured fitness: attenuation of regression coefficients and of the additive variance
# y = x'alpha + e, (x, y) jointly Gaussian; keep individuals with y > c (keep fraction q).
# delta = lambda(lambda - c') with c' = standardized cut, lambda = phi(c')/q.
# slope on survivors: alpha (1-delta)/(1 - delta R2); s2A on survivors with survivor genotype variance:
# ratio_surv = (1-delta)^2/(1-delta R2);  with population genotype variance (rescaled): (1-delta)^2/(1-delta R2)^2
import numpy as np
from scipy.stats import norm
def delta(q):
    c = norm.ppf(1-q); lam = norm.pdf(c)/q
    return lam*(lam-c)
print("q    delta   | R2=0.1: slope  s2A_surv s2A_resc | R2=0.5 | R2=0.65")
for q in [0.9, 0.75, 0.5, 0.25]:
    d = delta(q); row = f"{q:4.2f} {d:.4f} |"
    for R2 in [0.1, 0.5, 0.65]:
        s = (1-d)/(1-d*R2)
        row += f" {s:.3f} {(1-d)**2/(1-d*R2):.3f} {s*s:.3f} |"
    print(row)
# Monte Carlo check: L = 20 independent loci, large n, OLS on survivors
rng = np.random.default_rng(3)
L, n = 200, 400000
p = rng.uniform(0.1, 0.9, L)
G = rng.binomial(2, p, size=(n, L)).astype(float)
X = G - 2*p
a = rng.standard_normal(L)*0.05
g = X @ a
s2A = a @ np.diag(2*p*(1-p)) @ a
for R2 in [0.5, 0.65]:
    se = np.sqrt(s2A*(1-R2)/R2)
    y = g + rng.standard_normal(n)*se
    keep = y > np.median(y)
    Xs = X[keep]-X[keep].mean(0); ys = y[keep]-y[keep].mean()
    ah = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    Vs = np.cov(Xs.T)
    d = delta(0.5)
    print(f"R2={R2}: slope ratio MC {np.sum(ah*a)/np.sum(a*a):.3f} theory {(1-d)/(1-d*R2):.3f};",
          f"s2A surv MC {ah@Vs@ah/s2A:.3f} theory {(1-d)**2/(1-d*R2):.3f};",
          f"rescaled MC {ah@np.diag(2*p*(1-p))@ah/s2A:.3f} theory {((1-d)/(1-d*R2))**2:.3f}")
    # truncated regression MLE style correction with known delta and estimated R2 on survivors? 
    # invert: alpha = ahat (1-delta R2)/(1-delta); R2 is unknown but identifiable from survivor phenotype variance:
    vy_s = ys.var(); vy = vy_s/(1-d)        # population phenotypic variance from survivors
    s2A_s = ah @ Vs @ ah                    # = (1-d)^2/(1-dR2) s2A
    # solve for s2A: s2A_s = (1-d)^2 s2A / (1 - d s2A/vy)
    s2A_corr = s2A_s/((1-d)**2 + d*s2A_s/vy)
    print("   corrected s2A ratio", s2A_corr/s2A)
