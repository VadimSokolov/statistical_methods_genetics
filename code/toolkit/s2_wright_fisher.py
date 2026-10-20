"""Section 2: Wright Fisher and Moran fixation probabilities, drift."""
import numpy as np
from scipy import stats

def wf_matrix(N, s):
    i = np.arange(N+1); p = i/N
    pp = p*(1+s)/(1+s*p)
    return stats.binom.pmf(i[None,:], N, pp[:,None])

def fix_prob(N, s):
    P = wf_matrix(N, s)
    Q = P[1:N,1:N]; R = P[1:N, N]
    u = np.linalg.solve(np.eye(N-1)-Q, R)
    return u   # u[i-1] = fixation prob from i copies

def kimura(p, N, s):
    if s == 0: return p
    return (1-np.exp(-2*N*s*p))/(1-np.exp(-2*N*s))

def moran(i, N, s):
    r = 1+s
    return (1-r**(-i))/(1-r**(-N))

print("N s exactWF kimura moran 2s/(1-e^{-2Ns})...")
rows = []
for N in [10, 20, 50]:
    for s in [0.01, 0.05, 0.1]:
        u = fix_prob(N, s)[0]
        k = kimura(1/N, N, s)
        m = moran(1, N, s)
        print(N, s, round(u,4), round(k,4), round(m,4), "rel err %.1f%%" % (100*(k-u)/u))
# neutral check
print("neutral N=20", fix_prob(20,0.0)[:3])
# p=1/2 with selection
print("N=20,s=0.05,p=1/2 exact", round(fix_prob(20,0.05)[9],4), "kimura", round(kimura(0.5,20,0.05),4))
# Haldane 2s for large N
N=1000; s=0.01
print("N=1000 s=.01 kimura", kimura(1/N,N,s), "2s/(1+s)?", 2*s, "1-e^{-2s}", 1-np.exp(-2*s))
# mean absorption time, neutral, from p=1/2
for N in [10,20,50]:
    P = wf_matrix(N, 0.0); Q = P[1:N,1:N]
    t = np.linalg.solve(np.eye(N-1)-Q, np.ones(N-1))
    p = 0.5
    print("N", N, "exact T(1/2)", round(t[N//2-1],3), "diffusion", round(-2*N*(p*np.log(p)+(1-p)*np.log(1-p)),3))
# heterozygosity decay check: E[2X(1-X)] after t gens, N=20 from p=1/2
N=20; P = wf_matrix(N,0.0); v = np.zeros(N+1); v[10]=1
x = np.arange(N+1)/N
for t in range(1,11):
    v = v@P
    if t in (1,5,10):
        H = v@(2*x*(1-x))
        print("t",t,"H",round(H,5),"pred",round(0.5*(1-1/N)**t,5))
# Moran exact versus Kimura with N replaced by N/2
for N in [10,20,50]:
    for s in [0.01,0.05,0.1]:
        m = moran(1,N,s); k2 = (1-np.exp(-N*s/N))/(1-np.exp(-N*s))
        print("Moran", N, s, round(m,4), "Kimura(N/2)", round(k2,4), "rel %.1f%%" % (100*(k2-m)/m))
