import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln
def EK(t,n): return np.sum(t/(t+np.arange(n)))
def ll(t,data): return sum(k*np.log(t)+gammaln(t)-gammaln(t+n) for n,k in data)
def analyse(data,prior):
    K=sum(k for n,k in data)
    th=brentq(lambda t: sum(EK(t,n) for n,k in data)-K,1e-6,1e4)
    I=sum(np.sum(np.arange(n)/(th+np.arange(n))**2) for n,k in data)/th
    g=np.linspace(1e-4,60 if K<100 else 400,600001)
    lp=np.array([ll(t,data) for t in [g]])[0]+np.log(prior(g))
    w=np.exp(lp-lp.max());w/=w.sum();c=np.cumsum(w)
    return th,1/np.sqrt(I),(g*w).sum(),g[np.searchsorted(c,.5)],g[np.searchsorted(c,.025)],g[np.searchsorted(c,.975)]
poch=lambda t:1/((t+1)*(t+2))
d1=[(10,4)];print(analyse(d1,poch))
th=analyse(d1,poch)[0]
for t in [.5,1,5]: print(t,np.exp(ll(t,d1)-ll(th,d1)))
d3=[(10,4),(20,6),(15,3)];print(analyse(d3,poch))
print(analyse(d3,lambda t:np.exp(-0.1*t)))
dc=[(3306,501)];r=analyse(dc,poch);print(r)
th=r[0];n=3306
print(th*n/(th+n-1), th/2*n*(n-1)/((th+n-1)*(th+n-2)), np.sum(th/(th+np.arange(n,2*n))))
