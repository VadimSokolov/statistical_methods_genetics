"""Checks for Section "An arbitrary weight" of chapter py (round 2 fixes py_b-20, py_b-21, py_b-22).

Run:  python3 -I code/py_rev/arbitrary_weight.py      (about two minutes; output cached in
      code/py_rev/arbitrary_weight.txt)

Parts
  A  Proposition index needs more than regular variation of G.  A weight with G(r)/r^2 -> 1 that is
     flat except on very short intervals near the points of a grid with cell ratio 1 - eps -> 1 (a density vanishing on intervals) gives a
     counting function that grows like a power of log(1/x), not like x^(-1/3).  Compared with the
     smooth weight G(r) = r^2 on the same sticks.
  B  Proposition index under the density hypothesis, G(r) = r^2/(1 - log(r)/2), sigma = 1/2,
     theta = 1, size biased layout: the increment ratio p'_k / {rho G(R_{k-1}) B_k} tends to one, and
     the counting function N(x) is matched by the Gamma sandwich mean
     Lambda(x) = sum_k P(Gamma(1-sigma) > c x phi(k)), c = sigma/(rho M^rho), phi(k) = k^(1/s*)/l(k^-beta).
  C  Lemma karlin at gamma = 1 (regime (i)): G(r) = 1/(1 - log(r)/2), sigma = 1/2, theta = 1.
     E K(t) = sum_k (1 - exp(-t p_k)) against sum_k min(t p_k, 1) = t * int_0^{1/t} N(x) dx, with
     the mass beyond the last simulated stick added analytically (it is all in frequencies far below
     1/t).  Also n / E K_n against log(n)/2.
  D  sigma = 0 (py_b-20): G(r) = 1/(1 - log(r)/2), theta = 1.  E K_n / sqrt(n) against the
     prediction Gamma(1/2) Gamma(3/2) sqrt(2 theta) = 2.2214.
"""
import os
import sys
import numpy as np
from scipy.special import gamma as Gf, gammaincc

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
rng = np.random.default_rng(20261020)


def logR_sizebiased(sig, th, K):
    k = np.arange(1, K + 1)
    B = rng.beta(1 - sig, th + k * sig)
    return B, np.concatenate([[0.0], np.cumsum(np.log1p(-B))])


def EK(p, ts, tailmass=0.0):
    return np.array([np.sum(-np.expm1(-t * p)) + t * tailmass for t in ts])


def Nx(p, xs):
    ps = np.sort(p)
    return len(ps) - np.searchsorted(ps, xs, side='right')


print('=== A: regular variation of G alone is not enough (sigma=1/2, rho=2) ===')
sig, th, rho = 0.5, 1.0, 2.0
K = 10_000_000
B, L = logR_sizebiased(sig, th, K)
# grid r_{j+1} = r_j (1 - eps_j), eps_j = 1/sqrt(2 + log(1/r_j)); G rises only on
# [r_j(1-w_j), r_j] with w_j = min(eps_j^2, r_j^1.5), far narrower than one stick at that scale
lr = [0.0]
while lr[-1] > L[-1] - 5:
    e = 1 / np.sqrt(2 - lr[-1])
    lr.append(lr[-1] + np.log1p(-e))
lr = np.array(lr)  # decreasing log grid
eps = 1 / np.sqrt(2 - lr[:-1])


def G_flat(logr):
    # cell j: logr in (lr[j+1], lr[j]]
    j = np.searchsorted(-lr, -logr, side='left') - 1
    j = np.clip(j, 0, len(lr) - 2)
    top, bot = lr[j], lr[j + 1]
    w = np.minimum(eps[j] ** 2, np.exp(1.5 * top))
    rise_lo = top + np.log1p(-w)
    Gtop, Gbot = np.exp(rho * top), np.exp(rho * bot)
    frac = np.clip((logr - rise_lo) / (top - rise_lo), 0, 1)
    return Gbot + (Gtop - Gbot) * frac


Gf_vals = G_flat(L)
r = np.exp(L)
print('max |G_flat(r)/r^2 - 1| over r in (1e-6,1e-5): %.3f, over r < 1e-6: %.3f' % (
    np.max(np.abs(Gf_vals / r ** 2 - 1)[(r > 1e-6) & (r < 1e-5)]), np.max(np.abs(Gf_vals / r ** 2 - 1)[r < 1e-6])))
p_flat = -np.diff(Gf_vals)
p_pow = -np.diff(r ** rho)
xs = 10.0 ** -np.arange(6, 15, 2)
print('x        N_flat   N_power   x^(1/3) N_power')
for x, a, b in zip(xs, Nx(p_flat, xs), Nx(p_pow, xs)):
    print('%.0e  %7d  %8d   %.3f' % (x, a, b, x ** (1 / 3) * b))
ts = 10.0 ** np.arange(4, 13, 2)
ek_f, ek_p = EK(p_flat, ts), EK(p_pow, ts)
print('t        EK_flat  EK_power  local slopes flat / power')
for i, t in enumerate(ts):
    print('%.0e  %8.1f  %8.1f' % (t, ek_f[i], ek_p[i]))
print('slopes flat ', np.round(np.diff(np.log(ek_f)) / np.diff(np.log(ts)), 3))
print('slopes power', np.round(np.diff(np.log(ek_p)) / np.diff(np.log(ts)), 3), '(sigma_* = 1/3)')

print('\n=== B: Proposition index with density hypothesis, G = r^2/(1 - log r/2) ===')
beta = (1 - sig) / sig
ss = sig / (rho * (1 - sig) + sig)
ell = lambda logr: 1 / (1 - 0.5 * logr)
logG = lambda logr: rho * logr + np.log(ell(logr))
G = np.exp(logG(L))
p = -np.diff(G)
ratio = p / (rho * G[:-1] * B)
for k0 in [10, 1000, 100000, 9_000_000]:
    print('k=%8d  p_k/(rho G(R_{k-1}) B_k) = %.4f' % (k0, ratio[k0 - 1]))
kk = np.arange(K - 1_000_000, K + 1)
M = np.exp(np.median(L[kk] + beta * np.log(kk)))
gs = rng.gamma(1 - sig, size=400_000)
c = sig / (rho * M ** rho)
k = np.arange(1, K + 1)
logphi = np.log(k) / ss - np.log(ell(-beta * np.log(k)))
xs = 10.0 ** -np.arange(8, 15, 2)
print('M = %.4f' % M)
for x in xs:
    # Lambda(x) = sum_k P(G > c x phi(k))
    lam = np.sum(gammaincc(1 - sig, np.exp(np.log(c * x) + logphi)))
    print('x=%.0e  N(x) = %8d  Lambda(x) = %10.1f  ratio %.4f' % (x, Nx(p, [x])[0], lam, Nx(p, [x])[0] / lam))

print('\n=== C: regime (i), Karlin at gamma = 1, G = 1/(1 - log r/2), sigma = 1/2 ===')
ts = 10.0 ** np.arange(3, 9)
for rep in range(3):
    B, L = logR_sizebiased(sig, th, 30_000_000)
    a, b = L[:-1], L[1:]
    p = 0.5 * (a - b) / ((1 - 0.5 * a) * (1 - 0.5 * b))
    tail = 1 / (1 - 0.5 * L[-1])
    ek = EK(p, ts, tail)
    mn = np.array([np.sum(np.minimum(t * p, 1)) + t * tail for t in ts])
    print('rep %d, tail mass beyond 3e7 sticks %.3f, max t p_K %.1e' % (rep, tail, ts[-1] * p[-1]))
    print('  t      EK(t)/sum min(tp,1)   t/EK(t)   log(t)/2')
    for i, t in enumerate(ts):
        print('  %.0e   %.4f              %.3f     %.3f' % (t, ek[i] / mn[i], t / ek[i], np.log(t) / 2))

print('\n=== D: sigma = 0, G = 1/(1 - log r/2), theta = 1 ===')
ts = 10.0 ** np.arange(2, 9)
vals = []
for rep in range(20):
    B, L = logR_sizebiased(0.0, 1.0, 4_000_000)
    a, b = L[:-1], L[1:]
    p = 0.5 * (a - b) / ((1 - 0.5 * a) * (1 - 0.5 * b))
    tail = 1 / (1 - 0.5 * L[-1])
    vals.append(EK(p, ts, tail) / np.sqrt(ts))
vals = np.array(vals)
print('prediction Gamma(1/2)Gamma(3/2)sqrt(2) = %.4f' % (Gf(0.5) * Gf(1.5) * np.sqrt(2)))
for t, m, s in zip(ts, vals.mean(0), vals.std(0) / np.sqrt(len(vals))):
    print('n=%.0e  E K_n / sqrt(n) = %.4f +- %.4f' % (t, m, s))
