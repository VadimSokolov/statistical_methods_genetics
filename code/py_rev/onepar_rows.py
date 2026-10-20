"""Check the 'image' column of Table py:tab:onepar at n = 2.

Under ESF(t) two sampled points share an interval with probability 1/(1+t).
For each row, the coincidence probability E[sum_k p_k^2] of the induced
frequencies is estimated by averaging sum_k p_k^2 over GEM(theta) stick
realisations (truncated at 200 sticks; omitted mass below 1e-17 in mean).
If K_n ~ c log n the only candidate ESF law is ESF(c), so a row is
'not ESF' at n = 2 when its coincidence probability differs from 1/(1+c).

Usage: python3 -I onepar_rows.py [reps] [seed]
"""
import sys
import numpy as np

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20261020
NST = 200
rng = np.random.default_rng(SEED)

for theta in (1.0, 2.0):
    for alpha in (0.5, 2.0, 3.0):
        B = rng.beta(1.0, theta, size=(REPS, NST))
        R = np.cumprod(1 - B, axis=1)
        Rprev = np.hstack([np.ones((REPS, 1)), R[:, :-1]])
        rows = {
            "1 uniform, G=r": (Rprev - R, theta),
            "2/3 size biased, G=r^a": (Rprev**alpha - R**alpha, theta / alpha),
            "4 size biased left, G=1-(1-r)^a": ((1 - (1 - Rprev)**alpha) - (1 - (1 - R)**alpha), theta),
        }
        # row 5: ranked layout, G = r^a
        V = np.sort(Rprev - R, axis=1)[:, ::-1]
        tail = 1 - np.cumsum(V, axis=1)
        tail = np.clip(tail, 0, None)
        tprev = np.hstack([np.ones((REPS, 1)), tail[:, :-1]])
        rows["5 ranked, G=r^a"] = (tprev**alpha - tail**alpha, theta / alpha)
        for name, (p, c) in rows.items():
            q = (p**2).sum(axis=1)
            m, se = q.mean(), q.std(ddof=1) / np.sqrt(REPS)
            target = 1 / (1 + c)
            print(f"theta={theta} alpha={alpha} row {name:34s} P(coincide)={m:.5f} +- {se:.5f}"
                  f"  ESF({c:g}) value={target:.5f}  z={(m - target) / se:+.1f}")
        print()
