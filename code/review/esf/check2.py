import numpy as np
from scipy.special import digamma, polygamma, gammaln
from scipy.optimize import brentq
a = np.array([118,74,44,24,29,22,20,19,20,15,12,14,6,12,6,9,9,6,10,10,11,5,3,3]); j=np.arange(1,25)
n=3306; K=501
def pyEK(al,t,n): return (t/al)*(np.exp(gammaln(t+al+n)-gammaln(t+al)-gammaln(t+n)+gammaln(t))-1)
al,th=-0.49359507,362.8798263
print("pyEK n",pyEK(al,th,n),"2n",pyEK(al,th,2*n),"new from 501",pyEK(al,th,2*n)-501,"diff",pyEK(al,th,2*n)-pyEK(al,th,n))
# EK via sum of EA
def pyEA(al,t,jj,n):
    return np.exp(gammaln(n+1)-gammaln(jj+1)-gammaln(n-jj+1)+np.log(t+al)+gammaln(jj-al)-gammaln(1-al)+gammaln(t+al+n-jj)-gammaln(t+al+1)-gammaln(t+n)+gammaln(t+1))
print("sumEA 2n", pyEA(al,th,np.arange(1,2*n+1),2*n).sum())
EK=lambda t,n=n: t*(digamma(t+n)-digamma(t))
ll=lambda t: K*np.log(t)+gammaln(t)-gammaln(t+n)
t=K/np.log(n); out=[t]
for i in range(7):
    s=K/t-(digamma(t+n)-digamma(t)); J=K/t**2-(polygamma(1,t)-polygamma(1,t+n)); t=t+s/J; out.append(t)
print("newton",np.round(out,3))
th=brentq(lambda u:EK(u)-K,1,2000); lm=ll(th)
print("LR",brentq(lambda u:ll(u)-lm+1.92,50,th),brentq(lambda u:ll(u)-lm+1.92,th,1000))
# posterior quadrature Gamma(1,0.01)
g=np.linspace(80,300,200001); lp=ll(g)-0.01*g; w=np.exp(lp-lp.max()); w/=w.sum()
c=np.cumsum(w); print("post mean",(g*w).sum(),"median",g[np.searchsorted(c,.5)],"5%",g[np.searchsorted(c,.05)],"95%",g[np.searchsorted(c,.95)], "sd", np.sqrt((g**2*w).sum()-((g*w).sum())**2))
