"""Section 1: Hardy Weinberg, linkage disequilibrium, average effects."""
import numpy as np
from scipy import stats

# Ford's scarlet tiger moth (Panaxia dominula), Cothill 1940 sample
nAA, nAa, naa = 1469, 138, 5
n = nAA + nAa + naa
p = (2*nAA + nAa)/(2*n); q = 1-p
exp = n*np.array([p*p, 2*p*q, q*q])
obs = np.array([nAA, nAa, naa])
chi2 = ((obs-exp)**2/exp).sum()
print("moth n", n, "p", round(p,4), "expected", exp.round(1), "chi2", round(chi2,3),
      "pval(1df)", round(1-stats.chi2.cdf(chi2,1),3))
Fhat = 1 - (nAa/n)/(2*p*q)
print("Fhat", round(Fhat,4))

# Linkage disequilibrium example
h = dict(AB=0.5, Ab=0.1, aB=0.1, ab=0.3)
pA = h['AB']+h['Ab']; pB = h['AB']+h['aB']
D = h['AB']-pA*pB
D2 = h['AB']*h['ab']-h['Ab']*h['aB']
r2 = D**2/(pA*(1-pA)*pB*(1-pB))
Dmax = min(pA*(1-pB), (1-pA)*pB)
print("pA pB", pA, pB, "D", round(D,4), "check", round(D2,4), "r2", round(r2,4), "D'", round(D/Dmax,4))
c = 0.1
for t in [1,5,10,20]:
    print(" D_t c=0.1 t=%d %.5f" % (t, D*(1-c)**t))
# simulate recursion of haplotype freqs under random mating with recombination c
x = np.array([h['AB'],h['Ab'],h['aB'],h['ab']])
for t in range(10):
    Dt = x[0]*x[3]-x[1]*x[2]
    x = x + c*Dt*np.array([-1,1,1,-1])
print(" after 10 gens D", round(x[0]*x[3]-x[1]*x[2],5), "pred", round(D*0.9**10,5))

# One locus average effects
p, q = 0.4, 0.6
G = np.array([10., 8., 4.])     # A1A1, A1A2, A2A2
f = np.array([p*p, 2*p*q, q*q])
X = np.array([2., 1., 0.])      # copies of A1
mu = f@G
a1 = p*G[0]+q*G[1]-mu; a2 = p*G[1]+q*G[2]-mu
alpha = a1-a2
a = (G[0]-G[2])/2; d = G[1]-(G[0]+G[2])/2
print("mean", mu, "a1", a1, "a2", a2, "alpha", alpha, "a+d(q-p)", a+d*(q-p))
W = np.diag(f); Z = np.column_stack([np.ones(3), X])
beta = np.linalg.solve(Z.T@W@Z, Z.T@W@G)
print("WLS intercept slope", beta)
BV = np.array([2*a1, a1+a2, 2*a2])
dev = G-mu-BV
print("breeding values", BV, "dominance dev", dev.round(4))
VG = f@(G-mu)**2; VA = f@BV**2; VD = f@dev**2
print("VG", VG, "VA", VA, "2pq alpha^2", 2*p*q*alpha**2, "VD", VD, "(2pqd)^2", (2*p*q*d)**2)
print("h2 narrow", VA/VG)

# inbreeding F: average excess vs average effect
F = 0.2
fF = np.array([p*p+F*p*q, 2*p*q*(1-F), q*q+F*p*q])
muF = fF@G
# average excess of A1: mean genotypic value of carriers of a random A1 gene minus mean
P11 = fF[0]+fF[1]/2
ex1 = (fF[0]*G[0] + fF[1]/2*G[1])/p - muF
ex2 = (fF[2]*G[2] + fF[1]/2*G[1])/q - muF
Wf = np.diag(fF)
bF = np.linalg.solve(Z.T@Wf@Z, Z.T@Wf@G)
print("F=0.2 mean", muF, "excess diff", ex1-ex2, "effect (WLS slope)", bF[1], "ratio", (ex1-ex2)/bF[1])
