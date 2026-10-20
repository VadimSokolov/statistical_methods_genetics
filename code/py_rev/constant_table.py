"""Recheck Table py:tab:constant and the sentence after it.

For each row (sigma, theta, alpha) draw PD(sigma, theta) sticks, form the
unweighted frequencies W_k and the weighted ones R_{k-1}^alpha - R_k^alpha on the
same realisation, count N(x) and N_*(x) at level x, and record
Z_*/Z^kappa with Z = x^sigma N(x), Z_* = x^sigma_* N_*(x).  The prediction is
C Gamma(1-sigma)^kappa / Gamma(1-sigma_*) with C from eq. py:eq:C.

Usage: python3 -I constant_table.py [reps] [sticks] [seed]
"""
import sys
import numpy as np
from scipy.special import gamma

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
K = int(float(sys.argv[2])) if len(sys.argv) > 2 else 2_000_000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 20261020
X = 1e-7
ROWS = [(0.5, 1, 2), (0.4, 1, 2), (0.6, 1, 1.5), (0.5, 1, 1), (0.5, 2, 2)]


def sstar(s, a):
    return s / (a * (1 - s) + s)


def predicted(s, a):
    ss = sstar(s, a)
    kap = a * ss / s
    C = a**ss * s**((a - 1) * ss) * gamma(1 - ss) * gamma(1 - s + ss) / gamma(1 - s)
    return C * gamma(1 - s)**kap / gamma(1 - ss)


rng = np.random.default_rng(SEED)
k = np.arange(1, K + 1)
out = {}
for (s, t, a) in ROWS:
    ss = sstar(s, a)
    kap = a * ss / s
    r = np.empty(REPS)
    for i in range(REPS):
        B = rng.beta(1 - s, t + k * s)
        logR = np.cumsum(np.log1p(-B))
        logRprev = np.concatenate(([0.0], logR[:-1]))
        W = np.exp(logRprev) * B
        Wst = np.exp(a * logRprev) * (-np.expm1(a * np.log1p(-B)))
        Z = X**s * np.count_nonzero(W > X)
        Zs = X**ss * np.count_nonzero(Wst > X)
        r[i] = Zs / Z**kap
    m, sd = r.mean(), r.std(ddof=1)
    out[(s, t, a)] = (m, sd)
    print(f"sigma={s} theta={t} alpha={a} sigma*={ss:.4f} predicted={predicted(s, a):.4f} "
          f"simulated={m:.4f} relsd={sd / m:.3f} se_mean={sd / np.sqrt(REPS):.4f} "
          f"(sim-pred)/pred={(m / predicted(s, a) - 1) * 100:+.2f}%")

print("\nTable values: theta=1 row 1.3624, theta=2 row 1.3450, prediction 1.3661")
print(f"table change theta 1 to 2: {(1.3450 / 1.3624 - 1) * 100:+.2f}%")
print(f"table theta=2 vs prediction: {(1.3450 / 1.3661 - 1) * 100:+.2f}%")
for (rs, pred) in [((0.051, 1.3624), 1.3661), ((0.083, 1.1726), 1.1623), ((0.015, 1.4094), 1.4058),
                   ((0.041, 1.3450), 1.3661)]:
    se = rs[0] * rs[1] / np.sqrt(60)
    print(f"table sim {rs[1]} pred {pred}: diff {(rs[1] - pred) / se:+.2f} standard errors of the mean")
se1 = 0.051 * 1.3624 / np.sqrt(60)
se5 = 0.041 * 1.3450 / np.sqrt(60)
print(f"table theta 1 vs 2 difference in standard errors (independent): "
      f"{(1.3624 - 1.3450) / np.hypot(se1, se5):.2f}")
m1, m5 = out[(0.5, 1, 2)][0], out[(0.5, 2, 2)][0]
print(f"this run: change theta 1 to 2: {(m5 / m1 - 1) * 100:+.2f}%")
