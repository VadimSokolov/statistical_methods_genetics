import numpy as np
from scipy.special import gammaln
g=np.geomspace(1e-2,400,4000)
lp0=-np.log((g+1)*(g+2))
# use trapezoid weights on geometric grid
dg=np.gradient(g)
def post(k,n):
    lp=lp0+k*np.log(g)+gammaln(g)-gammaln(g+n)
    w=np.exp(lp-lp.max())*dg; w/=w.sum(); c=np.cumsum(w)
    return (g*w).sum(), g[np.searchsorted(c,.025)], g[np.searchsorted(c,.975)]
def lawK(th,n):
    p=np.array([1.0])
    for j in range(n):
        q=th/(th+j); p=np.concatenate([p*(1-q),[0]])+np.concatenate([[0],p*q])
    return p
for th,n in [(5,100),(5,1000),(1,100)]:
    P=lawK(th,n); ks=np.arange(len(P))
    res=[post(k,n) if k>0 else (0,0,0) for k in ks]
    lo=np.array([r[1] for r in res]); hi=np.array([r[2] for r in res])
    cov=P[(lo<=th)&(hi>=th)].sum()
    def wmed(x):
        o=np.argsort(x); c=np.cumsum(P[o]); return x[o][np.searchsorted(c,.5)]
    print(th,n,'median lo hi',wmed(lo),wmed(hi),'median width/th',wmed(hi-lo)/th,'cov',cov)
from scipy.optimize import brentq
k,n=15,100
print(brentq(lambda t: np.sum(t/(t+np.arange(n)))-k,.01,100), post(15,100))
