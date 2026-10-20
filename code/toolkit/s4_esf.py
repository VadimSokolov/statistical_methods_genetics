"""Section 4: Ewens and Pitman formulae, n=4 enumeration, K_n moments, Fisher's alpha."""
import numpy as np
from fractions import Fraction as Fr
from math import factorial, prod
from scipy import optimize, special

def rising(x, n):
    r = Fr(1) if isinstance(x, Fr) else 1.0
    for i in range(n): r *= (x+i)
    return r

def a_vectors(n):
    # partitions of n as multiplicity vectors
    def parts(n, m):
        if n == 0: yield []; return
        for k in range(min(n, m), 0, -1):
            for rest in parts(n-k, k): yield [k]+rest
    for p in parts(n, n):
        a = [0]*n
        for k in p: a[k-1] += 1
        yield tuple(a), p

def esf(a, th):
    n = sum((j+1)*x for j, x in enumerate(a))
    v = Fr(factorial(n))/rising(th, n)
    for j, x in enumerate(a):
        v *= (th/Fr(j+1))**x / factorial(x)
    return v

def pitman_a(a, blocks, al, th):
    # probability of the multiplicity vector a: EPPF times number of set partitions of that type
    n = sum(blocks); k = len(blocks)
    eppf = prod([th + i*al for i in range(1, k)]) / rising(th+1, n-1)
    for b in blocks: eppf *= rising(1-al, b-1)
    mult = Fr(factorial(n))
    for j, x in enumerate(a): mult /= (factorial(j+1)**x)*factorial(x)
    return eppf*mult

n = 4
for th in [Fr(1), Fr(2)]:
    tot = 0; EK = 0
    print("theta", th)
    for a, p in a_vectors(n):
        pr = esf(a, th); tot += pr; EK += pr*sum(a)
        print("  ", p, a, pr, float(pr))
    print("  total", tot, "EK", EK, float(EK), "formula", sum(th/(th+i) for i in range(n)))
    VK = sum(th*i/(th+i)**2 for i in range(n))
    EK2 = sum(esf(a,th)*sum(a)**2 for a,_ in a_vectors(n))
    print("  VarK", EK2-EK**2, "formula", VK)
al, th = Fr(1,2), Fr(1)
tot = 0; EK = 0
print("Pitman alpha=1/2 theta=1")
for a, p in a_vectors(n):
    pr = pitman_a(a, p, al, th); tot += pr; EK += pr*sum(a)
    print("  ", p, pr, float(pr))
print("  total", tot, "EK", EK, float(EK), "formula", (th/al)*(rising(th+al, n)/rising(th, n)-1))
# Pitman reduces to Ewens at alpha=0
print("check alpha=0", all(pitman_a(a,p,Fr(0),Fr(2))==esf(a,Fr(2)) for a,p in a_vectors(6)))

# CRP simulation check of K_4 law, theta=2
rng = np.random.default_rng(3)
R = 400000
K = np.ones(R, int)
for i in range(1, n):
    K += rng.random(R) < 2/(2+i)
print("sim P(K=k), theta=2:", [round((K==k).mean(),4) for k in range(1,5)])
print("exact:", [float(sum(esf(a,Fr(2)) for a,_ in a_vectors(4) if sum(a)==k)) for k in range(1,5)])
# Stirling numbers
print("unsigned Stirling s(4,k):", [int(abs(x)) for x in [6,11,6,1]])

# Williams Macrolepidoptera n=15609, K=240
N, S = 15609, 240
fa = optimize.brentq(lambda a: a*np.log(1+N/a)-S, 1, 500)
x = N/(N+fa)
print("Fisher alpha", round(fa,3), "x", round(x,6), "expected singletons alpha*x", round(fa*x,2))
EKf = lambda t: t*(special.digamma(t+N)-special.digamma(t))
mle = optimize.brentq(lambda t: EKf(t)-S, 1, 500)
se = 1/np.sqrt((EKf(mle) - mle**2*(special.polygamma(1,mle)-special.polygamma(1,mle+N)))/mle**2)
print("ESF MLE", round(mle,3), "SE", round(se,3))
# expected a_j under ESF at mle vs logseries for j=1,2,3
for j in [1,2,3,10]:
    ej = mle/j*np.exp(special.gammaln(N+1)-special.gammaln(N-j+1)+special.gammaln(mle+N-j)-special.gammaln(mle+N))
    print(" j", j, "ESF E a_j", round(ej,3), "logseries", round(fa*x**j/j,3))
print("theta log n", mle*np.log(N), "E K", EKf(mle))
