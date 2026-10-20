"""Effective size table of Section suff:sec:ess (round 3: the values had no script).

Separate sexes, N_m males and N_f females, each offspring has a father and a mother drawn
uniformly with replacement. With theta_t the coancestry of two distinct individuals in generation
t and F_t = theta_{t-1} the inbreeding coefficient,
    theta_t = (1/4) [ c (1 + theta_{t-2}) / 2 + (2 - c) theta_{t-1} + 2 theta_{t-1} ],  c = 1/N_m + 1/N_f,
which for N_m = N_f = 1 is the brother sister recursion F_t = (1 + 2 F_{t-1} + F_{t-2}) / 4.
With q_t = 1 - F_t (probability not yet identical by descent), the inbreeding size over k
generations is N_k = 1 / (2 (1 - (q_{k+1}/q_1)^{1/k})), counted from the first generation
with F > 0 (founders F_{-1} = F_0 = 0); the convention reproduces the values printed in the text, the eigenvalue size is
1 / (2 (1 - lambda_1)), and the geometric rate is |lambda_2 / lambda_1|.
Run: python3 -I ess_table.py   (prints and writes ess_table.json beside this file)
"""
import json
from pathlib import Path
import numpy as np


def run(Nm, Nf, ks=(1, 2, 5, 10, 20, 50)):
    c = 1 / Nm + 1 / Nf
    # q recursion: q_t = (c/8) q_{t-2} + ((4 - c)/4) q_{t-1}
    lam = np.roots([1, -(4 - c) / 4, -c / 8])
    lam = lam[np.argsort(-np.abs(lam))]
    q = [1.0, 1.0]                      # F_{-1} = F_0 = 0 (unrelated founders)
    for _ in range(max(ks) + 2):
        q.append(c * q[-2] / 8 + (4 - c) * q[-1] / 4)
    q = np.array(q)[2:]                 # q[0] = q_1, the first generation with F > 0
    Nk = {k: 1 / (2 * (1 - (q[k] / q[0]) ** (1 / k))) for k in ks}
    Ne = 1 / (2 * (1 - lam[0].real))
    return {"N_k": Nk, "N_e_eigen": Ne, "rate": abs(lam[1] / lam[0]),
            "excess_times_k": {k: k * (Nk[k] - Ne) for k in ks}}


out = {"Nm1_Nf1": run(1, 1), "Nm2_Nf20": run(2, 20)}
print(json.dumps(out, indent=1, default=float))
(Path(__file__).resolve().parent / "ess_table.json").write_text(json.dumps(out, indent=1, default=float))
