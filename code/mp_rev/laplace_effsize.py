"""Effective model size E m_n at tau_n = n^(-1/2) for priors outside the Be(a,b) family.

Reproduces the statement after Corollary growth in chapters/mp.tex (round 2, item mp_b-03):
for the Bayesian lasso (lambda^2 ~ Exp(1)) and a fixed scale ridge prior (lambda = 1) the
expected effective model size E m_n = n E[tau^2 lambda^2 / (1 + tau^2 lambda^2)] with
tau^2 = 1/n is bounded (it tends to E lambda^2 = 1), whereas the Be(1,1) prior grows like
log n (Theorem effsize, Table effsize). Exact one dimensional quadrature with mpmath.

Run: python3 -I code/mp_rev/laplace_effsize.py   (output cached in laplace_effsize.txt)
"""
import mpmath as mp

mp.mp.dps = 30


def em_lasso(n):
    # lambda^2 ~ Exp(1): E m_n = n * int_0^inf (s/n)/(1+s/n) e^{-s} ds
    t2 = mp.mpf(1) / n
    return n * mp.quad(lambda s: t2 * s / (1 + t2 * s) * mp.e ** (-s), [0, 1, 10, mp.inf])


def em_ridge(n):
    # lambda = 1 fixed: E m_n = n * (1/n)/(1+1/n) = n/(n+1)
    return mp.mpf(n) / (n + 1)


def em_beta(n, a, b):
    # kappa_1 ~ Be(a,b), kappa_tau = 1/(1+tau^2 lambda^2) with lambda^2 = (1-k)/k
    t2 = mp.mpf(1) / n
    f = lambda k: (t2 * (1 - k) / k) / (1 + t2 * (1 - k) / k) * k ** (a - 1) * (1 - k) ** (b - 1) / mp.beta(a, b)
    return n * mp.quad(f, [0, mp.mpf(1) / n, mp.mpf(1) / mp.sqrt(n), 1])


if __name__ == "__main__":
    print("n        lasso Exp(1)   ridge lambda=1   Be(1,1)     log n")
    for e in [2, 4, 6, 8]:
        n = 10 ** e
        print("1e%d  %12.4f  %14.4f  %10.3f  %8.2f" % (
            e, float(em_lasso(n)), float(em_ridge(n)), float(em_beta(n, 1, 1)), float(mp.log(n))))
