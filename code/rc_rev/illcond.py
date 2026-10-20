"""Digits lost by the residue expansion in double precision.
Kernel: Ewens likelihood theta^k/[theta]^n times prior 1/((theta+1/2)(theta+3/2)),
i.e. PH(0,1/2,2,1), chosen so that every pole is simple. Normalizer by residues in float64
against the same formula in 60 digit arithmetic and against quadrature."""
import numpy as np
import mpmath as mp

def kernel_poles(n):
    # theta^k/[theta]^n = theta^(k-1)/prod_{i=1}^{n-1}(theta+i); prior poles 1/2, 3/2
    return [mp.mpf(i) for i in range(1, n)] + [mp.mpf(1) / 2, mp.mpf(3) / 2]

def residues(n, k, prec_dps):
    with mp.workdps(prec_dps):
        r = kernel_poles(n)
        g = []
        for i, ri in enumerate(r):
            num = (-ri) ** (k - 1)
            den = mp.fprod([rl - ri for l, rl in enumerate(r) if l != i])
            g.append(num / den)
        return r, g

def normalizer_mp(n, k, dps):
    with mp.workdps(dps):
        r, g = residues(n, k, dps)
        return -mp.fsum(gi * mp.log(ri) for gi, ri in zip(g, r)), max(abs(x) for x in g), mp.fsum(g)

def normalizer_double(n, k):
    r = np.array([float(i) for i in range(1, n)] + [0.5, 1.5])
    g = np.empty(len(r))
    for i in range(len(r)):
        d = np.delete(r, i) - r[i]
        g[i] = (-r[i]) ** (k - 1) / np.prod(d)
    return -np.sum(g * np.log(r)), np.abs(g).max(), np.sum(g)

print("n k  exact C  double C  rel err  digits correct  log10 kappa  sum gamma (double)")
rows = []
for n in [10, 15, 20, 25, 30, 35, 40]:
    k = 3
    Cex, gmax, s = normalizer_mp(n, k, 120)
    Cd, gmd, sd = normalizer_double(n, k)
    rel = abs(Cd - float(Cex)) / float(Cex)
    kap = float(mp.log10(gmax / Cex))
    # quadrature check
    with mp.workdps(30):
        f = lambda x: x ** (k - 1) / mp.fprod([x + i for i in range(1, n)]) / ((x + 0.5) * (x + 1.5))
        q = mp.quad(f, [0] + [mp.mpf(2) ** j for j in range(-6, 14)] + [mp.inf], maxdegree=10)
    print(n, k, mp.nstr(Cex, 12), "%.6e" % Cd, "%.2e" % rel,
          "%.1f" % (-np.log10(rel) if rel > 0 else 16), "%.2f" % kap, "%.1e" % sd,
          " quad", mp.nstr(q, 12))
    rows.append((n, k, Cex, Cd, rel, kap))
