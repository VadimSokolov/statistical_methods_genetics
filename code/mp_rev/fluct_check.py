"""Check of the fluctuation table (K_n under PD(1/2,0) vs horseshoe m_n), 400 replicates.
Run: python -I fluct_check.py"""
import numpy as np
from scipy.special import gammaln
rng = np.random.default_rng(11)
R = 400; al = 0.5
for n in [200, 1000, 5000]:
    K = np.zeros(R)
    for m in range(n):
        p = np.ones(R) if m == 0 else (al*K)/m
        K += rng.random(R) < p
    kap = rng.beta(0.5, 0.5, (R, n)); lam2 = (1-kap)/kap; t2 = 1.0/n
    mm = np.sum(t2*lam2/(1+t2*lam2), 1)
    exactK = np.exp(gammaln(n+al)-np.log(al)-gammaln(al)-gammaln(n))
    print(n, "K mean %.2f sd %.2f cv %.3f exact mean %.2f" % (K.mean(), K.std(ddof=1), K.std(ddof=1)/K.mean(), exactK),
          "| m mean %.2f sd %.2f cv %.3f pred cv %.3f" % (mm.mean(), mm.std(ddof=1), mm.std(ddof=1)/mm.mean(), 2**-0.5*n**-0.25))
