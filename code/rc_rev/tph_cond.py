"""Is the tilted normalizer with numerator prod_i [alpha]^{n_i} ill conditioned?
Prior PH(0,1/2,2,1), K=40 counts from NB(1.7,1/2), lambda = K log 2."""
import numpy as np, mpmath as mp
rng = np.random.default_rng(1)
nn = rng.negative_binomial(1.7, 0.5, size=40)
N = nn.sum(); lam = 40 * np.log(2)
print("N", N)
def Z(dps):
    with mp.workdps(dps):
        a = mp.mpf(1) / 2
        P = [mp.mpf(1)]  # coefficients, ascending
        for n in nn:
            for j in range(n):
                # multiply by (alpha + j)
                newP = [mp.mpf(0)] * (len(P) + 1)
                for k, c in enumerate(P):
                    newP[k] += c * j; newP[k + 1] += c
                P = newP
        Pf = lambda x: mp.fsum(c * x ** k for k, c in enumerate(P))
        r = [a, a + 1]
        g1 = Pf(-r[0]) / (r[1] - r[0]); g2 = Pf(-r[1]) / (r[0] - r[1])
        # polynomial part: divide P by (x+r0)(x+r1) synthetic
        q = P[:]
        def divide(coefs, root):  # divide by (x + root), ascending coefs
            d = len(coefs) - 1
            out = [mp.mpf(0)] * d
            rem = coefs[d]
            for k in range(d - 1, -1, -1):
                out[k] = rem
                rem = coefs[k] - root * rem
            return out, rem
        q1, _ = divide(q, r[0]); q2, _ = divide(q1, r[1])
        L = mp.mpf(lam)
        terms = [mp.factorial(k) * c / L ** (k + 1) for k, c in enumerate(q2)]
        res = [g * mp.exp(L * ri) * mp.e1(L * ri) for g, ri in zip((g1, g2), r)]
        tot = mp.fsum(terms) + mp.fsum(res)
        mx = max(abs(x) for x in terms + res)
        return tot, mx
t1, m1 = Z(200)
t2, m2 = Z(16)
with mp.workdps(30):
    f = lambda x: mp.exp(-lam * x + mp.fsum(mp.loggamma(x + n) - mp.loggamma(x) for n in nn)) / ((x + 0.5) * (x + 1.5))
    q = mp.quad(f, [0, 0.5, 1, 2, 4, 8, 16, mp.inf])
print("exact", mp.nstr(t1, 15), "16 digits", mp.nstr(t2, 15), "quad", mp.nstr(q, 15), "log10 kappa", mp.nstr(mp.log10(m1 / t1), 4))
for d in [16, 20, 25, 30, 40]:
    print(d, mp.nstr(Z(d)[0], 15))
