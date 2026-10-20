"""Shared helpers for the coal chapter revision: exact law of S and a full coalescent simulator."""
import numpy as np
from fractions import Fraction
from math import comb


def a(n, r=1):
    return sum(1.0 / j**r for j in range(1, n))


def pmf_S(n, theta, smax):
    """Exact pmf of S by convolving geometric pmfs (Theorem geom)."""
    out = np.zeros(smax + 1); out[0] = 1.0
    s = np.arange(smax + 1)
    for k in range(2, n + 1):
        p = (k - 1) / (k - 1 + theta)
        g = p * (1 - p) ** s
        out = np.convolve(out, g)[: smax + 1]
    return out


def pmf_S_tavare(n, theta, s):
    """Classical closed form: sum_{i=2}^n (-1)^i C(n-1,i-1) (i-1)/(theta+i-1) (theta/(theta+i-1))^s."""
    return sum((-1) ** i * comb(n - 1, i - 1) * (i - 1) / (theta + i - 1) * (theta / (theta + i - 1)) ** s
               for i in range(2, n + 1))


def pmf_S_frac(n, theta, s):
    """Exact rational pmf for rational theta via complete homogeneous polynomial."""
    theta = Fraction(theta)
    ps = [Fraction(k - 1) / (k - 1 + theta) for k in range(2, n + 1)]
    qs = [1 - p for p in ps]
    # coefficient of z^s in prod 1/(1-q z): dynamic programming
    coef = [Fraction(1)] + [Fraction(0)] * s
    for q in qs:
        new = [Fraction(0)] * (s + 1)
        acc = Fraction(0)
        for j in range(s + 1):
            acc = acc * q + coef[j]
            new[j] = acc
        coef = new
    prod = Fraction(1)
    for p in ps:
        prod *= p
    return prod * coef[s]


def simulate(n, theta, rng, return_sites=False):
    """One coalescent with infinite sites mutation. Returns dict with S_k, xi, H, sites (bitmasks)."""
    lineages = [1 << i for i in range(n)]
    Sk = {}
    sites = []
    for k in range(n, 1, -1):
        T = rng.exponential(2.0 / (k * (k - 1)))
        m = rng.poisson(theta * T / 2, size=k)
        Sk[k] = int(m.sum())
        for lin, c in zip(lineages, m):
            sites.extend([lin] * int(c))
        i, j = rng.choice(k, size=2, replace=False)
        merged = lineages[i] | lineages[j]
        lineages = [l for t, l in enumerate(lineages) if t not in (i, j)] + [merged]
    xi = np.zeros(n, dtype=int)
    for b in sites:
        xi[bin(b).count("1")] += 1
    rows = set()
    for leaf in range(n):
        rows.add(tuple(((b >> leaf) & 1) for b in sites))
    out = dict(Sk=Sk, S=len(sites), xi=xi[1:], H=len(rows))
    if return_sites:
        out["sites"] = sites
    return out


def ewens_EK(n, theta):
    return sum(theta / (theta + i - 1) for i in range(1, n + 1))
