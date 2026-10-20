"""Two parameter partition law for n = 4: exact probabilities and a Dirichlet multinomial check;
one illustrative step of the Beta augmentation sampler for the butterflies."""
from fractions import Fraction as Fr
from math import factorial
from itertools import product
import numpy as np


def partitions(n, maxpart=None):
    if maxpart is None:
        maxpart = n
    if n == 0:
        yield []
        return
    for p in range(min(n, maxpart), 0, -1):
        for rest in partitions(n - p, p):
            yield [p] + rest


def rising(x, m):
    r = Fr(1)
    for i in range(m):
        r *= x + i
    return r


def nsetpart(shape):
    n = sum(shape); c = Fr(factorial(n))
    for j in set(shape):
        a = shape.count(j); c /= Fr(factorial(j)) ** a * factorial(a)
    return c


def py(shape, al, th):
    k = len(shape); n = sum(shape)
    num = Fr(1)
    for i in range(1, k):
        num *= th + i * al
    for nj in shape:
        num *= rising(1 - al, nj - 1)
    return nsetpart(shape) * num / rising(th + 1, n - 1)


for al, th in ((Fr(1, 2), Fr(1)), (Fr(-1), Fr(2)), (Fr(0), Fr(1))):
    tot = 0; EK = 0
    print("alpha", al, "theta", th)
    for s in partitions(4):
        p = py(s, al, th); tot += p; EK += p * len(s)
        print("  ", "+".join(map(str, s)), nsetpart(s), p, round(float(p), 4))
    print("  sum", tot, "EK", EK, float(EK))

# direct check alpha = -1, theta = 2: two types, uniform frequency P ~ Beta(1,1)
# P(counts c, 4-c in a given order) = int p^c (1-p)^(4-c) dp = c!(4-c)!/5!
probs = {}
for seq in product((0, 1), repeat=4):
    c = sum(seq)
    pr = Fr(factorial(c) * factorial(4 - c), factorial(5))
    shape = tuple(sorted([x for x in (c, 4 - c) if x > 0], reverse=True))
    probs[shape] = probs.get(shape, 0) + pr
print("direct two types:", probs)

# Beta augmentation, one illustrative step at theta = 164 for the butterflies
n, k, a, b = 3306, 501, 1.0, 0.01
th = 164.0
eta_mean = (th + 1) / (th + 1 + n)
c = b - np.log(eta_mean)
odds = (a + k - 1) / (n * c)
pi = odds / (1 + odds)
print("E eta", eta_mean, "c", c, "pi", pi, "mixture mean", pi * (a + k) / c + (1 - pi) * (a + k - 1) / c)
