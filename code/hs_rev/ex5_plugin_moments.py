# Mean and variance of the plug in under orthogonal design; bilinear bias; Monte Carlo checks
import numpy as np
L, n, s2, s2A = 400, 300, 0.0325, 0.0213
bias = L*s2/n
var = 4*s2*s2A/n + 2*L*s2**2/n**2
print("old values (0.0325, 0.0213): bias", bias, "E plug", s2A+bias, "ratio", (s2A+bias)/s2A, "sd", np.sqrt(var), "sd/s2A", np.sqrt(var)/s2A)
# Values quoted after Proposition hs:prop:bias (round 3): phenotypic variance 0.0320 and sigma^2_A 0.0205,
# both from code/hs_rev/sim_main_check.txt (summary.json run)
_s2, _s2A = 0.0320, 0.0205
_b = L*_s2/n; _v1 = 4*_s2*_s2A/n; _v2 = 2*L*_s2**2/n**2
print("text values (0.0320, 0.0205): bias %.4f  E plug/truth %.3f  sd %.4f  sd/s2A %.3f  share of var from 2L s^4/n^2 %.3f"
      % (_b, (_s2A+_b)/_s2A, np.sqrt(_v1+_v2), np.sqrt(_v1+_v2)/_s2A, _v2/(_v1+_v2)))
# MC check with an orthogonal design (Hadamard-like), small case
rng = np.random.default_rng(11)
L, n = 40, 64
from scipy.linalg import hadamard
H = hadamard(n)[:, 1:L+1].astype(float)         # orthogonal centred columns, x'x = n
h = rng.uniform(0.2, 0.5, L)
X = H*np.sqrt(h)                                  # x_l'x_l = n h_l
a = np.zeros(L); a[:5] = rng.standard_normal(5)*0.1
s2 = 0.04; s2A = np.sum(h*a*a)
M = 200000
E = rng.standard_normal((M, n))*np.sqrt(s2)
Ah = (E @ X)/(n*h) + a
P = (h*Ah**2).sum(1)
print("MC mean", P.mean(), "theory", s2A + L*s2/n)
print("MC var", P.var(), "theory", 4*s2*s2A/n + 2*L*s2**2/n**2)
# bilinear
b = np.zeros(L); b[:5] = rng.standard_normal(5)*0.5; rho, sw, sz = 0.5, 0.2, 1.0
Ew = rng.standard_normal((M, n)); Ez = rho*Ew + np.sqrt(1-rho**2)*rng.standard_normal((M, n))
Ah = (sw*Ew @ X)/(n*h) + a; Bh = (sz*Ez @ X)/(n*h) + b
B = (h*Ah*Bh).sum(1)
print("bilinear MC", B.mean(), "theory", np.sum(h*a*b) + L*rho*sw*sz/n)
print("text example bias L=300 n=250 rho=.5 sw=.1 sz=1:", 300*0.5*0.1*1/250)
