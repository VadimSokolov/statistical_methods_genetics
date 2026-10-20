"""Horseshoe under a dense truth (Sections ftns:sec:horseshoe, 'What it is not', and ftns:sec:omnigenic).

Every one of L = 2000 deviations is nonzero, delta_l ~ N(0, sd^2) in units of its standard error,
y_l = delta_l + N(0,1). tau is fitted by marginal maximum likelihood and the posterior total
sum_l E[delta_l^2 | y] is compared with the true sum_l delta_l^2 (ftns_rev_lib.fit). 20 replicates
per sd, seed 1. Small dense deviations are shrunk away (the total is badly underestimated); dense
deviations of a few standard errors give a large tau hat and little shrinkage.

Run: python3 -I code/ftns_rev/check_dense.py   (output cached in check_dense.txt)
"""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np
from ftns_rev_lib import fit, simulate
rng = np.random.default_rng(1)
for sd in [0.3, 0.5, 1.0, 2.0, 4.0]:
    r = []
    for rep in range(20):
        d, y = simulate(rng, 2000, 1.0, sd); P = fit(y); r.append((P['tau'], P['total']/np.sum(d*d)))
    r = np.array(r)
    print(f"sd={sd:3.1f}  mean tau hat={r[:,0].mean():.3f}  mean retained fraction of the total={r[:,1].mean():.3f}")
