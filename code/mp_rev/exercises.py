"""Numbers quoted in the hints to the exercises of chapter mp. Run: python -I exercises.py"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln
d = np.array([6., 4., 4.]); b = np.array([1., .5, 0.]); N = 6
E = lambda l: (np.sum(d*(d/(d+l))**2*b**2) + np.sum((d/(d+l))**2))/N
lam = brentq(lambda l: E(l) - 7/6, 0, 2); print("ex3 lambda with unbiased sigma_A^2:", lam, E(lam))
p = (1 - np.sqrt(0.5))/2; print("ex2 argmax p", p)
def eppf(sizes, al, th):
    k = len(sizes); n = sum(sizes)
    num = np.prod([th+i*al for i in range(1, k)]) if k > 1 else 1.0
    den = np.exp(gammaln(th+n)-gammaln(th+1))
    return num/den*np.prod([np.exp(gammaln(m-al)-gammaln(1-al)) for m in sizes])
mult = {(4,): 1, (3, 1): 4, (2, 2): 3, (2, 1, 1): 6, (1, 1, 1, 1): 1}
tot = 0
for s, m in mult.items():
    v = eppf(s, 0.5, 0); tot += m*v; print("ex6", s, v, m*v)
print("total", tot)
from itertools import product
pr = [1, 1/2, 1/3, 1/4]; dist = np.zeros(5)
for bits in product([0, 1], repeat=4):
    if bits[0] == 0: continue
    q = np.prod([pr[j] if bits[j] else 1-pr[j] for j in range(4)]); dist[sum(bits)] += q
print("ex9 K_4 law theta=1:", dist*24)
th = 5.0
I = lambda n: np.sum(np.arange(n)/(th+np.arange(n))**2)/th
print("ex11 se at n=100", 1/np.sqrt(I(100)), "n=1e4", 1/np.sqrt(I(10**4)), "n=1e6", 1/np.sqrt(I(10**6)))
for L, r in [(100, 0.1)]: print("ex12 factor", 1+(L-1)*r)
