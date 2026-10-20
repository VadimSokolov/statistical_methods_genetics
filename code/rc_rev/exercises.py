"""Numerical checks of the answers given in the exercise hints."""
import mpmath as mp
from math import comb
mp.mp.dps = 30
hh = lambda x: 1 / ((x + 1) * (x + 2) * mp.log(2))
print("E a^-1/2", mp.quad(lambda x: x ** -0.5 * hh(x), [0, 1, mp.inf]), mp.pi * (1 - 1 / mp.sqrt(2)) / mp.log(2))
print("E a^1/2", mp.quad(lambda x: x ** 0.5 * hh(x), [0, 1, mp.inf]), mp.pi * (mp.sqrt(2) - 1) / mp.log(2))
# n=4, K=3
f = lambda x: x ** 2 / ((x + 1) ** 2 * (x + 2) ** 2 * (x + 3))
C = mp.quad(f, [0, 1, mp.inf]); M = mp.quad(lambda x: x * f(x), [0, 1, 10, mp.inf])
print("K=3: C", C, 2.5 - 2.25 * mp.log(3), "mean", M / C)
print("P(new) K=3", mp.quad(lambda x: x / (x + 4) * f(x), [0, 1, 10, mp.inf]) / C)
# tilt check
lam = mp.mpf('0.01')
print("tilt", mp.quad(lambda x: mp.exp(-lam * x) / ((x + 1) * (x + 2)), [0, 1, 100, mp.inf]),
      mp.exp(lam) * mp.e1(lam) - mp.exp(2 * lam) * mp.e1(2 * lam), mp.log(2))
# k=1 : smallest n with log10 kappa_n >= 16
mp.mp.dps = 60
for n in range(44, 60):
    J = mp.quad(lambda x: mp.gamma(x + 1) * mp.gamma(n) / mp.gamma(x + n), [0, 0.1, 1, 10, mp.inf])
    kap = comb(n - 2, (n - 2) // 2) * (n - 1) / J
    if mp.log10(kap) >= 16:
        print("first n with kappa >= 1e16:", n, mp.log10(kap)); break
# Gamma(2a)/Gamma(a)^2 duplication check
a = mp.mpf('0.7')
print("dup", mp.gamma(2 * a) / mp.gamma(a) ** 2, 2 ** (2 * a - 1) * mp.gamma(a + 0.5) / (mp.sqrt(mp.pi) * mp.gamma(a)))
# mode of theta/((t+1)^2 (t+2)^2 (t+3)): roots of the score
import numpy as np
from numpy.polynomial import polynomial as P
t = np.poly1d([1, 0])
num = (t+1)*(t+2)*(t+3) - 2*t*(t+2)*(t+3) - 2*t*(t+1)*(t+3) - t*(t+1)*(t+2)
print("score numerator", num, "roots", num.roots)
