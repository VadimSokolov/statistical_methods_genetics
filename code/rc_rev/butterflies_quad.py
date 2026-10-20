# NOTE (round 3): this script reproduces the UNCENSORED analysis (the 24 tabulated classes taken
# as the whole sample, n = 3306, K = 501, theta hat 163.99), which chapter rc now quotes only as the
# "ignoring the censoring" comparison (Section on Corbet's butterflies, Example rc:ex:butterflies).
# The censored analysis the chapter uses (n hat = 11068, K = 620) is in butterflies_censored.py
# and butterflies_censored_residue.py.
"""Fine quadrature check of the butterfly normalizer and mean under PH(0,1,2,1)."""
import mpmath as mp
mp.mp.dps = 50
n, K = 3306, 501
g = lambda t: mp.exp(K * mp.log(t) + mp.loggamma(t) - mp.loggamma(t + n)) / ((t + 1) * (t + 2))
pts = [0, 20, 50, 80, 100, 110, 120] + list(range(125, 260, 5)) + [300, 400, 600, 1000, mp.inf]
C = mp.quad(g, pts, maxdegree=12)
M = mp.quad(lambda t: t * g(t), pts, maxdegree=12)
P = mp.quad(lambda t: t / (t + n) * g(t), pts, maxdegree=12)
print("C", mp.nstr(C, 15), "mean", mp.nstr(M / C, 15), "P(new)", mp.nstr(P / C, 15))
