"""Exact enumeration of allelic partitions for small n under the Ewens sampling formula.
Checks normalisation, E K_n, Var K_n, E A_j, Stirling numbers, Hoppe urn path probability."""
from fractions import Fraction as Fr
from math import factorial, comb


def partitions(n, maxpart=None):
    if maxpart is None:
        maxpart = n
    if n == 0:
        yield []
        return
    for p in range(min(n, maxpart), 0, -1):
        for rest in partitions(n - p, p):
            yield [p] + rest


def rising(t, n):
    r = Fr(1)
    for i in range(n):
        r *= (t + i)
    return r


def count_perm(shape):
    n = sum(shape)
    c = Fr(factorial(n))
    for j in set(shape):
        a = shape.count(j)
        c /= Fr(j) ** a * factorial(a)
    return c


def esf(shape, th):
    th = Fr(th)
    return count_perm(shape) * th ** len(shape) / rising(th, sum(shape))


for n in (3, 4, 5):
    print("n =", n)
    for th in (Fr(1), Fr(2), Fr(1, 2)):
        tot = Fr(0); EK = Fr(0); EK2 = Fr(0)
        EA = [Fr(0)] * (n + 1)
        rows = []
        for s in partitions(n):
            p = esf(s, th)
            tot += p; EK += p * len(s); EK2 += p * len(s) ** 2
            for j in range(1, n + 1):
                EA[j] += p * s.count(j)
            rows.append((s, count_perm(s), p))
        VK = EK2 - EK ** 2
        EKf = sum(th / (th + i) for i in range(n))
        VKf = sum(th * i / (th + i) ** 2 for i in range(n))
        EAf = [None] + [Fr(comb(n, j)) * factorial(j - 1) * th * rising(th, n - j) / rising(th, n) for j in range(1, n + 1)]
        print(" theta", th, "sum", tot, "EK", EK, float(EK), "check", EK == EKf,
              "VarK", VK, float(VK), "check", VK == VKf, "EA ok", EA[1:] == EAf[1:])
        print("  EA", [str(x) for x in EA[1:]], [round(float(x), 4) for x in EA[1:]])
        for s, c, p in rows:
            print("   ", "+".join(map(str, s)), c, p, round(float(p), 4))
    # Stirling numbers: P(K_n=k) = |s(n,k)| th^k / th_(n)
    st = {}
    for s in partitions(n):
        st[len(s)] = st.get(len(s), 0) + count_perm(s)
    print(" unsigned Stirling", st)

# Hoppe urn path: tables 1,1,2,1,3 for n = 5
for th in (Fr(1), Fr(2)):
    p = Fr(1) * 1 / (th + 1) * th / (th + 2) * 2 / (th + 3) * th / (th + 4)
    eppf = th ** 3 * factorial(2) / rising(th, 5)
    print("path prob", th, p, eppf, p == eppf)
# number of set partitions with shape 3+1+1 of {1..5}
print("set partitions 3+1+1:", factorial(5) // (factorial(3) * 2), "shape prob th=1:", esf([3, 1, 1], 1))

# homozygosity: E sum over pairs
for th in (1, 2):
    print("P(X1=X2)", Fr(1, th + 1))
