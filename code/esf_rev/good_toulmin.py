"""Good and Toulmin alternating sum on Corbet's table and its Ewens expectation."""
import numpy as np
from scipy.special import gammaln
a = np.array([118, 74, 44, 24, 29, 22, 20, 19, 20, 15, 12, 14, 6, 12, 6, 9, 9, 6, 10, 10, 11, 5, 3, 3])
j = np.arange(1, 25)
print("alternating sum", int(((-1.0) ** (j + 1) * a).sum()))
n, th = 3306, 163.99428
jj = np.arange(1, n + 1)
logEA = np.log(th) - np.log(jj) + gammaln(n + 1) - gammaln(n - jj + 1) + gammaln(th + n - jj) - gammaln(th + n)
EA = np.exp(logEA)
s = (-1.0) ** (jj + 1)
print("E alt sum all j", (s * EA).sum(), "j<=24", (s * EA)[:24].sum(), "sum EA", EA.sum())
i = np.arange(n, 2 * n)
print("E new", (th / (th + i)).sum())

# sampling distribution of the alternating sum (classes j <= 24) under the Ewens urn at theta = 164
rng = np.random.default_rng(7)
def urn_counts(thv, n):
    lab = np.empty(n, dtype=np.int64); k = 0
    u = rng.random(n); w = rng.random(n)
    for i in range(n):
        if u[i] < thv / (thv + i):
            lab[i] = k; k += 1
        else:
            lab[i] = lab[int(w[i] * i)]
    return np.bincount(lab)
G = []
for r in range(2000):
    c = urn_counts(164.0, n)
    c = c[c <= 24]
    G.append(((-1.0) ** (c + 1)).sum())
G = np.array(G)
print("sim alt sum mean", G.mean(), "sd", G.std(), "P(<=75)", (G <= 75).mean(), "q", np.quantile(G, [0.005, 0.05, 0.95]))
