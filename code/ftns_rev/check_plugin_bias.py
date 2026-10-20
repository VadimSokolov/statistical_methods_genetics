"""Check of the plug in bias 10.19/n of the estimated additive variance (Section ftns:sec:bias and the
verification table of Section ftns:app:num).

Setting (ftns.tex, viability assay): s_g ~ Bin(n, theta_g), theta = (0.45, 0.60, 0.35), p = 0.3 known,
c = (p, q-p, -q), alpha = c'theta = 0.13, sigma_A^2 = 2pq alpha^2 estimated by 2pq alpha_hat^2.
Because alpha_hat is unbiased, E[alpha_hat^2] = alpha^2 + Var(alpha_hat), so the relative bias is
exactly v/n with v = sum_g c_g^2 theta_g(1-theta_g) / alpha^2 = 10.186.
The Monte Carlo check uses 20000 replicates at each of n = 25, 50, 100 (seed 20261020). Its own
standard error is 2.4, 3.2 and 4.5 percent of the bias at these n, so the check can only confirm the
exact value to within that error; at larger n a 20000 replicate check resolves nothing.

Run: python3 -I code/ftns_rev/check_plugin_bias.py   (output cached in check_plugin_bias.txt)
"""
import numpy as np
th = np.array([0.45, 0.60, 0.35]); p = 0.3; q = 1-p
c = np.array([p, q-p, -q]); alpha = c @ th
v = (c**2 * th*(1-th)).sum() / alpha**2
print(f"alpha = {alpha:.4f}, exact relative plug in bias = {v:.4f}/n")
rng = np.random.default_rng(20261020)
R = 20000
for n in [25, 50, 100]:
    s = rng.binomial(n, th, size=(R, 3)) / n
    a = s @ c
    rel = a**2 / alpha**2 - 1
    est, se = rel.mean(), rel.std(ddof=1)/np.sqrt(R)
    print(f"n={n:4d}: MC relative bias {est:.5f} (MC s.e. {se:.5f}), exact {v/n:.5f}, "
          f"MC/exact - 1 = {est/(v/n)-1:+.4f}, MC s.e./exact = {se/(v/n):.4f}")
