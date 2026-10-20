"""Numbers for the revised chapter mp (worked examples and checks).
Run: python -I verify.py > verify_numbers.txt
"""
import numpy as np
from scipy.special import gammaln, betaln, digamma, gamma
from scipy.integrate import quad
from scipy.optimize import brentq

rng = np.random.default_rng(7)
print("== (a) optimism on a small design")
X = np.array([[1, 1, 0], [1, -1, 1], [1, 0, -1], [-1, 1, 1], [-1, -1, 0], [-1, 0, -1]], float)
N, p = X.shape
print("colsums", X.sum(0)); G = X.T @ X; print("X'X=\n", G)
print("eig X'X", np.round(np.linalg.eigvalsh(G), 4))
beta = np.array([1.0, 0.5, 0.0]); sig2 = 1.0
sA = np.sum((X @ beta)**2)/N; print("sigma_A^2 =", sA)
for lam in [0.0, 2.0, 6.0]:
    H = X @ np.linalg.solve(G + lam*np.eye(p), X.T)
    sig = np.sum((H @ X @ beta)**2)/N; tr = np.trace(H.T @ H); trH = np.trace(H)
    E = sig + sig2*tr/N
    y = X @ beta + rng.standard_normal((400000, N))
    mc = np.mean(np.sum((y @ H.T)**2, 1))/N
    print(f"lam={lam}: signal {sig:.4f} tr(H'H) {tr:.4f} trH {trH:.4f} E {E:.4f} MC {mc:.4f}")

print("== (b) three parameter beta normal: (a,b) -> GEM(1-a, a+b-1)")
def EK(n, al, th):
    # exact: Gamma(th+1)/(al Gamma(th+al)) Gamma(th+al+n)/Gamma(th+n) - th/al ; al=0: sum
    if al == 0:
        j = np.arange(n); return np.sum(th/(th+j))
    return np.exp(gammaln(th+1)-np.log(al)-gammaln(th+al)+gammaln(th+al+n)-gammaln(th+n)) - th/al
def EKrec(n, al, th):
    u = 1.0
    for m in range(1, n):
        u = u + (th+al*u)/(th+m)
    return u
def Ea1(n, a, b):
    return n*np.exp(betaln(a, b+n-1)-betaln(a, b))
def Em(n, a, b):
    t2 = 1.0/n
    f = lambda z: np.exp((a-1)*np.log(z)-(a+b)*np.log1p(z)-betaln(a, b))*t2/(t2+z)
    v = quad(f, 0, 1, limit=400, points=[t2])[0] + quad(f, 1, np.inf, limit=400)[0]
    return n*v
for (a, b, name) in [(0.5, 0.5, "horseshoe"), (1.0, 1.0, "uniform"), (0.5, 1.0, "Strawderman Berger")]:
    al, th = 1-a, a+b-1
    cnu = np.exp(gammaln(a+b)-gammaln(a)-gammaln(b))
    print(f"{name}: (a,b)=({a},{b}) alpha={al} theta={th} nu(x) = {cnu:.4f} x^{a-2}(1-x)^{b-1}")
    for n in [100, 1000, 10000]:
        ek = EK(n, al, th); er = EKrec(n, al, th)
        if al > 0:
            asy = np.exp(gammaln(th+1)-np.log(al)-gammaln(th+al))*n**al
        else:
            asy = th*np.log(n)
        print(f"  n={n}: EK {ek:.3f} (rec {er:.3f}) leading {asy:.3f} Ea1 {Ea1(n,a,b):.3f} Em_n {Em(n,a,b):.3f}")
    if al > 0:
        print("  PD const", np.exp(gammaln(th+1)-np.log(al)-gammaln(th+al)),
              " effsize const", np.pi/(np.exp(betaln(a, b))*np.sin(np.pi*a)))
# check E K_n = sum_j E a_j for horseshoe n=100
from scipy.special import comb
n = 100; a, b = 0.5, 1.0
s = sum(comb(n, j)*np.exp(betaln(a+j-1, b+n-j)-betaln(a, b)) for j in range(1, n+1))
print("sum_j E a_j at (1/2,1), n=100:", s, "EK", EK(100, 0.5, 0.5))

print("== (c) Kingman, n = 3")
def eppf(sizes, al, th):
    k = len(sizes); n = sum(sizes)
    num = np.prod([th+i*al for i in range(1, k)]) if k > 1 else 1.0
    den = np.exp(gammaln(th+1+n-1)-gammaln(th+1))
    return num/den*np.prod([np.exp(gammaln(m-al)-gammaln(1-al)) for m in sizes])
for al, th in [(0, 1), (0.5, 0), (0.5, 0.5), (0, 2)]:
    P = {s: eppf(s, al, th) for s in [(3,), (2, 1), (1, 1, 1)]}
    tot = P[(3,)] + 3*P[(2, 1)] + P[(1, 1, 1)]
    print(f"alpha={al} theta={th}: p(3)={P[(3,)]:.4f} p(2,1)={P[(2,1)]:.4f} p(1,1,1)={P[(1,1,1)]:.4f} total {tot:.4f}"
          f"  EK3 {P[(3,)]+6*P[(2,1)]+3*P[(1,1,1)]:.4f} vs {EK(3,al,th):.4f}")
# PK integral numerically, stable alpha=1/2
al = 0.5
psi = lambda t: t**al
kap = lambda m, t: al/gamma(1-al)*gamma(m-al)*t**(al-m)
for sizes in [(3,), (2, 1), (1, 1, 1)]:
    n = sum(sizes)
    f = lambda t: t**(n-1)*np.exp(-psi(t))*np.prod([kap(m, t) for m in sizes])/gamma(n)
    print("stable PK", sizes, quad(f, 0, np.inf)[0], eppf(sizes, al, 0))
th = 2.0
for sizes in [(3,), (2, 1), (1, 1, 1)]:
    n = sum(sizes)
    f = lambda t: t**(n-1)*(1+t)**(-th)*np.prod([th*gamma(m)*(1+t)**(-m) for m in sizes])/gamma(n)
    print("gamma PK", sizes, quad(f, 0, np.inf)[0], eppf(sizes, 0, th))

print("== (d) theta from a small sample")
GRID = np.geomspace(1e-3, 2000, 200001)
def post(ns, ks, prior="poch"):
    lp = np.zeros_like(GRID)
    for n, k in zip(ns, ks):
        lp += k*np.log(GRID) + gammaln(GRID) - gammaln(GRID+n)
    if prior == "poch":
        lp += -np.log(GRID+1) - np.log(GRID+2)
    else:
        c0, d0 = prior; lp += (c0-1)*np.log(GRID) - d0*GRID
    w = np.exp(lp-lp.max())*np.gradient(GRID); w /= w.sum()
    cdf = np.cumsum(w)
    q = lambda u: GRID[np.searchsorted(cdf, u)]
    return np.sum(w*GRID), q(0.5), q(0.025), q(0.975)
def mle(ns, ks):
    f = lambda t: sum(np.sum(t/(t+np.arange(n))) for n in ns) - sum(ks)
    return brentq(f, 1e-6, 1e6)
def info(ns, th):
    # Fisher information: Var K / th^2
    return sum(np.sum(th*np.arange(n)/(th+np.arange(n))**2) for n in ns)/th**2
for ns, ks in [([10], [4]), ([10, 20, 15], [4, 6, 3]), ([3306], [501])]:
    th = mle(ns, ks)
    print(f"n={ns} k={ks}: mle {th:.3f} se {1/np.sqrt(info(ns, th)):.3f}  poch post mean/median/95%:",
          np.round(post(ns, ks), 3))
# likelihood check small example
th = mle([10], [4]); print("check sum", np.sum(th/(th+np.arange(10))))
print("likelihood ratio L(theta)/L(mle) at 0.5,1,5:", [np.exp(4*np.log(t)+gammaln(t)-gammaln(t+10)-(4*np.log(th)+gammaln(th)-gammaln(th+10))) for t in [0.5, 1, 5]])
# Gibbs under Gamma(1, 0.1) prior vs quadrature, three loci
ns, ks = [10, 20, 15], [4, 6, 3]; c0, d0 = 1.0, 0.1
thv, out = 1.0, []
for it in range(60000):
    t = rng.beta(thv, ns)
    thv = rng.gamma(c0+sum(ks), 1.0/(d0-np.sum(np.log(t))))
    if it >= 1000: out.append(thv)
out = np.array(out)
print("Gibbs Ga(1,0.1): mean", out.mean().round(3), "q", np.quantile(out, [0.025, 0.5, 0.975]).round(3),
      " quadrature", np.round(post(ns, ks, (c0, d0)), 3))
# Corbet: predicted singletons and E K_2n
th = mle([3306], [501]); n = 3306
print("Corbet theta", th, "E a1", th*n/(th+n-1), "E a2", th/2*n*(n-1)/((th+n-1)*(th+n-2)),
      "E K_2n - K_n", np.sum(th/(th+np.arange(n, 2*n))))

print("== (e) effective size table check by quadrature")
for (a, b) in [(0.5, 0.5), (1, 1), (1, 3)]:
    print((a, b), [round(Em(n, a, b), 2) for n in [1000, 10000, 100000]],
          "pred", round(np.pi/(np.exp(betaln(a, b))*np.sin(np.pi*a))*1e5**(1-a), 2) if a < 1 else round(b*np.log(1e5), 2))
print("Gamma(2-a) min", gamma(1.461632), "CV ML alpha=1/2", np.sqrt(2*gamma(1.5)**2/gamma(2)-1))

print("== (f) log series check n=20 theta=2 and horseshoe spectrum")
from math import comb as C
n, th = 20, 2.0
Ea = [th/j*np.exp(gammaln(n+1)-gammaln(n-j+1)+gammaln(th+n-j)-gammaln(th+n)) for j in range(1, n+1)]
print("ESF E a_j j=1..6:", np.round(Ea[:6], 4), "sum j Ea_j", sum((j+1)*e for j, e in enumerate(Ea)), "sum Ea", sum(Ea), EK(n, 0, th))
Eh = [C(n, j)*np.exp(betaln(0.5+j-1, 0.5+n-j)-betaln(0.5, 0.5)) for j in range(1, n+1)]
print("HS E a_j j=1..6:", np.round(Eh[:6], 4), "sum j", sum((j+1)*e for j, e in enumerate(Eh)), "sum", sum(Eh), EK(n, 0.5, 0))
