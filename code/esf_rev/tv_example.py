"""Total variation between sampling k of n with and without replacement."""
from fractions import Fraction as Fr
from itertools import product


def p_nk(n, k):
    r = Fr(1)
    for i in range(k):
        r *= Fr(n - i, n)
    return 1 - r


def tv_values(xs, k):
    n = len(xs)
    vals = sorted(set(xs))
    cnt = {v: xs.count(v) for v in vals}
    tv = Fr(0)
    for tup in product(vals, repeat=k):
        pw = Fr(1)
        for v in tup:
            pw *= Fr(cnt[v], n)
        # without replacement
        c = dict(cnt); po = Fr(1); m = n
        for v in tup:
            po *= Fr(c[v], m); c[v] -= 1; m -= 1
        tv += abs(pw - po)
    return tv / 2


n = 10
for k in (2, 3, 4, 5):
    pk = p_nk(n, k)
    a = tv_values([0] * 6 + [1] * 4, k)
    b = tv_values(list(range(10)), k)
    print(k, "p", pk, float(pk), "bound", Fr(k * (k - 1), 2 * n), "TV 6/4:", a, round(float(a), 4),
          "TV distinct:", b, round(float(b), 4))

# number of red balls in k draws: hypergeometric vs binomial
from math import comb
for k in (2, 3, 4):
    tv = Fr(0)
    for r in range(k + 1):
        hyp = Fr(comb(4, r) * comb(6, k - r), comb(10, k))
        bi = Fr(comb(k, r)) * Fr(4, 10) ** r * Fr(6, 10) ** (k - r)
        tv += abs(hyp - bi)
    print("count TV", k, tv / 2, float(tv / 2))
# large n: k such that bound 0.05, and p_{n,k}
for n in (100, 1000, 10000):
    for k in (5, 10, 30):
        if k <= n:
            print(n, k, float(p_nk(n, k)), k * (k - 1) / (2 * n))
