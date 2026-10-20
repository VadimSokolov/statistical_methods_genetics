"""Cocycle gap of eq. py:eq:cocycle: sign and size pattern over sigma.

gap = C(sigma, a*a') / [C(T_{a'}(sigma), a) * C(sigma, a')^{kappa(T_{a'}(sigma), a)}] - 1,
with T_a(s) = s/(a(1-s)+s), kappa(s, a) = a/(a(1-s)+s) and C from eq. py:eq:C.
"""
import numpy as np
from scipy.special import gamma


def T(s, a):
    return s / (a * (1 - s) + s)


def kap(s, a):
    return a / (a * (1 - s) + s)


def C(s, a):
    ss = T(s, a)
    return a**ss * s**((a - 1) * ss) * gamma(1 - ss) * gamma(1 - s + ss) / gamma(1 - s)


def gap(s, a, b):
    return C(s, a * b) / (C(T(s, b), a) * C(s, b)**kap(T(s, b), a)) - 1


for (a, b) in [(2, 3), (3, 3), (4, 4), (0.5, 0.5), (4, 0.25), (0.25, 4), (2, 0.5), (0.5, 2)]:
    print(f"(a,a')=({a},{b}): " + "  ".join(f"s={s}: {100 * gap(s, a, b):+.3f}%"
                                         for s in (0.2, 0.4, 0.5, 0.65, 0.8)))

# sign pattern over a grid
S = np.linspace(0.02, 0.98, 49)
fade = [1.5, 2, 3, 4]
pers = [0.25, 0.5, 0.75]
both_fade = [gap(s, a, b) for s in S for a in fade for b in fade]
both_pers = [gap(s, a, b) for s in S for a in pers for b in pers]
mixed = [gap(s, a, b) for s in S for a in fade for b in pers] + \
        [gap(s, a, b) for s in S for a in pers for b in fade]
print("\nboth fade: min %+.4f max %+.4f" % (min(both_fade), max(both_fade)))
print("both persist: min %+.4f max %+.4f" % (min(both_pers), max(both_pers)))
print("mixed: min %+.4f max %+.4f" % (min(mixed), max(mixed)))
