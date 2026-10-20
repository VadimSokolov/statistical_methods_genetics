# Single locus worked example: average effects, variances, FTNS partial change, remainder, secondary theorem
import numpy as np
p = np.array([0.2, 0.8])
W = np.array([[1.10, 1.06], [1.06, 1.00]])
Z = np.array([[12.0, 11.0], [11.0, 9.0]])
def stats(p):
    wbar = p @ W @ p; wi = W @ p; alpha = wi - wbar
    zbar = p @ Z @ p; zi = Z @ p; beta = zi - zbar
    return wbar, wi, alpha, zbar, zi, beta
wbar, wi, alpha, zbar, zi, beta = stats(p)
s2A = 2*np.sum(p*alpha**2)
tot = np.sum(np.outer(p, p)*(W-wbar)**2)
# dominance = residual
fit = wbar + alpha[:, None] + alpha[None, :]
s2D = np.sum(np.outer(p, p)*(W-fit)**2)
dp = p*(wi-wbar)/wbar
pnew = p + dp
wbar2 = pnew @ W @ pnew
partial = 2*np.sum(alpha*dp)
print("wbar", wbar, "w_i", wi, "alpha", alpha)
print("s2A", s2A, "s2D", s2D, "total var", tot, "check", s2A+s2D)
print("dp", dp, "p'", pnew)
print("partial", partial, "s2A/wbar", s2A/wbar)
print("exact dw", wbar2-wbar, "R", wbar2-wbar-partial, "ratio", (wbar2-wbar)/partial)
# a = average effect of substitution
print("alpha1-alpha2 (subst effect)", alpha[0]-alpha[1], "h=2pq", 2*p[0]*p[1], "h*a^2", 2*p[0]*p[1]*(alpha[0]-alpha[1])**2)
# secondary
sAwz = 2*np.sum(p*alpha*beta)
zbar2 = pnew @ Z @ pnew
print("zbar", zbar, "z_i", zi, "beta", beta)
print("sigmaA(w,z)", sAwz, "partial dz", 2*np.sum(beta*dp), "sAwz/wbar", sAwz/wbar, "exact dz", zbar2-zbar)
