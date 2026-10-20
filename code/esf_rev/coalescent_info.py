"""Fisher information in K_n versus expected tree length / (2 theta), and the n = 4 tree."""
from fractions import Fraction as Fr
import numpy as np


def I(th, n):
    i = np.arange(n, dtype=float)
    return (i / (th * (th + i) ** 2)).sum()


def H(n):
    return (1.0 / np.arange(1, n)).sum()


for th in (0.5, 2.0, 10.0):
    for n in (4, 10, 100, 1000, 10**4, 10**6):
        a = I(th, n); b = H(n) / th
        print(th, n, round(a, 4), round(b, 4), round(b - a, 4), "ratio", round(a / b, 3))
    i = np.arange(1, 10**7, dtype=float)
    c = ((2 * i + th) / (i * (th + i) ** 2)).sum()
    print(" limit constant", c)

# n = 4 tree: E T_k = 2/(k(k-1)), E L = sum k E T_k
EL = sum(Fr(k) * Fr(2, k * (k - 1)) for k in range(2, 5))
VL = sum(Fr(k) ** 2 * Fr(2, k * (k - 1)) ** 2 for k in range(2, 5))
print("E L_4", EL, "Var L_4", VL, float(VL))
for th in (Fr(1), Fr(2)):
    In = sum(Fr(i) / (th * (th + i) ** 2) for i in range(4))
    print("theta", th, "I_4", In, float(In), "EL/(2th)", EL / (2 * th), float(EL / (2 * th)),
          "ratio", float(In / (EL / (2 * th))))
# killed coalescent n=3 shape probabilities
for th in (Fr(1), Fr(2)):
    p3 = Fr(2) / (th + 2) * 1 / (th + 1)
    p21 = th / (th + 2) * 1 / (th + 1) + Fr(2) / (th + 2) * th / (th + 1)
    p111 = th / (th + 2) * th / (th + 1)
    print(th, p3, p21, p111, p3 + p21 + p111)
