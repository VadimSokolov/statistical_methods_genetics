import numpy as np
from scipy.special import gammaln
a = np.array([118,74,44,24,29,22,20,19,20,15,12,14,6,12,6,9,9,6,10,10,11,5,3,3]); j=np.arange(1,25)
n=3306; K=501; i=np.arange(1,K)
def ll(al,t):
    v=t+i*al
    if np.any(v<=0): return -np.inf
    return np.sum(np.log(v))-(gammaln(t+n)-gammaln(t+1))+np.sum(a*(gammaln(j-al)-gammaln(1-al)))
A=np.linspace(-0.95,0.6,621); L=np.linspace(np.log(150),np.log(900),601)
M=np.array([[ll(x,np.exp(y)) for y in L] for x in A])
W=np.exp(M-M.max()); W/=W.sum()
pa=W.sum(1); pl=W.sum(0)
def q(v,p):
    c=np.cumsum(p); return (v*p).sum(), v[np.searchsorted(c,.05)], v[np.searchsorted(c,.95)]
print("alpha",q(A,pa)); T=np.exp(L); print("theta",q(T,pl))
mm=(np.exp(L)[None,:]/np.abs(A)[:,None]); neg=A<0
mv=mm[neg].ravel(); pw=W[neg].ravel(); o=np.argsort(mv); print("m",q(mv[o],pw[o]), "P(alpha<0)",W[neg].sum())
