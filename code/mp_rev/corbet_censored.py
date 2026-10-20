"""Example mp:ex:corbet with the censoring of Corbet's table (round 2). 620 species in all,
501 (3306 individuals) in the tabulated classes, 119 censored above 24 specimens; Chapter esf
profiles n out of the censored likelihood (theta hat 141.7474, n hat 11068, profile se 7.0;
code/esf_rev/butterflies_censored.py and butterflies_refit.txt). Given n the Ewens likelihood
is theta^620 Gamma(theta)/Gamma(theta+n). Prints the Pochhammer posterior at n hat, the Ewens
means of a1 and a2, the singleton fractions and the new species in a second sample of size n."""
import numpy as np
from scipy.special import gammaln
from scipy.optimize import brentq
from scipy.special import digamma

n, K, A1 = 11068, 620, 118
th = brentq(lambda t: t * (digamma(t + n) - digamma(t)) - K, 1, 5000)
info = sum((i - 1) / (th * (th + i - 1) ** 2) for i in range(1, n + 1))
print("theta hat %.4f  se given n %.3f (profile se over n: 7.0, Chapter esf)" % (th, info ** -0.5))
g = np.linspace(1e-6, 600, 600001)
lg = K * np.log(g) + gammaln(g) - gammaln(g + n) - np.log(g + 1) - np.log(g + 2)
w = np.exp(lg - lg.max()); cdf = np.cumsum(w); cdf /= cdf[-1]
print("Pochhammer posterior at n hat: mean %.2f 95%% [%.2f, %.2f]" %
      (np.sum(g * w) / w.sum(), g[np.searchsorted(cdf, 0.025)], g[np.searchsorted(cdf, 0.975)]))
Ea1 = th * n / (th + n - 1)
Ea2 = th / 2 * n * (n - 1) / ((th + n - 1) * (th + n - 2))
print("E a1 %.2f E a2 %.2f ; observed 118, 74" % (Ea1, Ea2))
print("singleton fraction observed 118/620 = %.3f, model E a1/K = %.3f ; (old 118/501 = %.3f)" %
      (A1 / K, Ea1 / K, 118 / 501))
print("E K_2n - K_n = %.2f" % np.sum(th / (th + np.arange(n, 2 * n))))
