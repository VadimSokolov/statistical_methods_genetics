"""Section 7: rejection ABC and quantile regression for the Ewens theta from K_n."""
import numpy as np
from scipy import special, optimize

rng = np.random.default_rng(17)
n, a0, b0 = 30, 1.0, 0.2
def simK(theta, rng):
    i = np.arange(n)
    return (rng.random((len(theta), n)) < theta[:, None]/(theta[:, None]+i[None, :])).sum(1)

grid = np.linspace(1e-5, 60, 600001)
def exact_post(K):
    lp = (K+a0-1)*np.log(grid) - b0*grid + special.gammaln(grid) - special.gammaln(grid+n)
    w = np.exp(lp-lp.max()); w /= w.sum(); return w
def exact_q(K, us):
    c = np.cumsum(exact_post(K)); return np.array([grid[np.searchsorted(c, u)] for u in us])

Kobs = 8
M = 200000
th = rng.gamma(a0, 1/b0, M); K = simK(th, rng)
acc = th[K == Kobs]
w = exact_post(Kobs)
print("ABC accept rate", round(len(acc)/M,4), "n acc", len(acc))
print("ABC mean sd", acc.mean().round(3), acc.std().round(3), "exact", (grid*w).sum().round(3),
      np.sqrt((grid**2*w).sum()-(grid*w).sum()**2).round(3))
us = [0.05, 0.5, 0.95]
print("ABC quantiles", np.quantile(acc, us).round(3), "exact", exact_q(Kobs, us).round(3))
# tolerance: accept |K-8|<=2
acc2 = th[np.abs(K-Kobs) <= 2]
print("ABC tol 2: rate", round(len(acc2)/M,4), "mean sd", acc2.mean().round(3), acc2.std().round(3))

# check loss minimiser over constants is the empirical quantile
z = rng.normal(size=1001)
rho = lambda r, u: r*(u-(r < 0))
for u in [0.1, 0.5, 0.9]:
    q = optimize.minimize_scalar(lambda c: rho(z-c, u).sum(), bounds=(-4,4), method="bounded").x
    print("u", u, "argmin", round(q,4), "empirical quantile", round(np.quantile(z, u, method="inverted_cdf"),4))

# quantile regression of log theta on (1, log K, (log K)^2) via linear programming
Ntr = 6000
tht = rng.gamma(a0, 1/b0, Ntr); Kt = simK(tht, rng)
y = np.log(tht); lk = np.log(Kt)
X = np.column_stack([np.ones(Ntr), lk, lk**2])
def qreg(u):
    p = X.shape[1]
    # min u*1'r+ + (1-u)*1'r-  s.t. X b + r+ - r- = y
    c = np.concatenate([np.zeros(2*p), u*np.ones(Ntr), (1-u)*np.ones(Ntr)])
    A = np.hstack([X, -X, np.eye(Ntr), -np.eye(Ntr)])
    res = optimize.linprog(c, A_eq=A, b_eq=y, bounds=(0, None), method="highs")
    b = res.x[:p]-res.x[p:2*p]; return b
for u in us:
    b = qreg(u)
    row = []
    for Kq in [3, 8, 15]:
        f = np.array([1, np.log(Kq), np.log(Kq)**2])
        row.append((Kq, round(float(np.exp(f@b)),3), round(float(exact_q(Kq,[u])[0]),3)))
    print("u", u, "fitted vs exact quantile at K=3,8,15:", row)
