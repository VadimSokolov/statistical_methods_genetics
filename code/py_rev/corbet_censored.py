"""Example py:ex:corbet with the censoring of Corbet's table (round 2). Fisher, Corbet and
Williams (1943) report 620 species; 501 (3306 individuals) lie in the tabulated classes
j = 1..24 and 119 are censored above 24. Chapter esf profiles n out of the censored Ewens
likelihood: theta hat = 141.7474, n hat = 11068 (code/esf_rev/butterflies_censored.py).
Prints the doubleton estimator, the observed ratio A2/A1 with its Poisson delta method se,
and the exact Ewens means E A1, E A2 at the censored fit (and at the old uncensored fit)."""
import numpy as np
from scipy.special import gammaln

C = np.array([118, 74, 44, 24, 29, 22, 20, 19, 20, 15, 12, 14, 6, 12, 6, 9, 9, 6, 10, 10, 11, 5, 3, 3])
A1, A2 = C[0], C[1]
print("tabulated K", C.sum(), "n", (np.arange(1, 25) * C).sum(), "1-2A2/A1 %.3f" % (1 - 2 * A2 / A1))
r = A2 / A1
se = r * np.sqrt(1 / A1 + 1 / A2)
print("A2/A1 %.3f  Poisson delta method se %.3f" % (r, se))


def EA(j, th, n):
    return th / j * np.exp(gammaln(n + 1) - gammaln(n - j + 1) + gammaln(th + n - j) - gammaln(th + n))


for th, n, lab in [(141.7474, 11068, "censored fit"), (163.99, 3306, "old, tabulated classes as whole sample")]:
    e1, e2 = EA(1, th, n), EA(2, th, n)
    print("%s: theta %.2f n %d E A1 %.2f E A2 %.2f ratio %.3f, (data - ratio)/se %.2f" %
          (lab, th, n, e1, e2, e2 / e1, (r - e2 / e1) / se))
