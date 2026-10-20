"""Reproduce the observed dataset of the ABC comparison (seed 8, theta = 8, first draw with S = 23)
using the simulator of code/suff/suff_p2.py, and report its spectrum, pi and Tajima's D."""
import numpy as np
n = 30
def sim_sfs(theta, rng):
    R = theta.size
    sizes = np.ones((R, n), int); xi = np.zeros((R, n + 1), int); rows = np.arange(R)
    for k in range(n, 1, -1):
        T = rng.exponential(1.0 / (k * (k - 1) / 2), R)
        mut = rng.poisson((theta * T / 2)[:, None] * np.ones((1, k)))
        np.add.at(xi, (np.repeat(rows, k), sizes[:, :k].ravel()), mut.ravel())
        i = rng.integers(0, k, R); j = rng.integers(0, k - 1, R); j = j + (j >= i)
        sizes[rows, i] += sizes[rows, j]; sizes[rows, j] = sizes[rows, k - 1]
    return xi[:, 1:n]
rng2 = np.random.default_rng(8)
tries = 0
while True:
    tries += 1
    xo = sim_sfs(np.array([8.0]), rng2)
    if xo.sum() == 23: break
xo = xo[0]
print("tries", tries, "spectrum", {i + 1: int(v) for i, v in enumerate(xo) if v})
i = np.arange(1, n)
S = xo.sum(); a1 = np.sum(1 / i); a2 = np.sum(1 / i**2)
pi = np.sum(i * (n - i) * xo) / (n * (n - 1) / 2)
b1 = (n + 1) / (3 * (n - 1)); b2 = 2 * (n * n + n + 3) / (9 * n * (n - 1))
c1 = b1 - 1 / a1; c2 = b2 - (n + 2) / (a1 * n) + a2 / a1**2
e1 = c1 / a1; e2 = c2 / (a1**2 + a2)
D = (pi - S / a1) / np.sqrt(e1 * S + e2 * S * (S - 1))
print("S", S, "theta_W", S / a1, "pi", pi, "Tajima D", D, "singletons", xo[0])
print("expected singletons at theta_W", S / a1)
