import numpy as np
from scipy import stats, special, integrate
from math import *
# t3 and laplace mixtures
b=1.3
f=lambda v: stats.norm.pdf(b,0,sqrt(v))*stats.invgamma.pdf(v,1.5,scale=1.5)
print('t3',integrate.quad(f,0,np.inf)[0],stats.t.pdf(b,3))
g=lambda v: stats.norm.pdf(b,0,sqrt(v))*stats.expon.pdf(v,scale=2)
print('lap',integrate.quad(g,0,np.inf)[0],0.5*exp(-1.3))
# horseshoe marginal
K=(2*pi**3)**-.5
def hs(be):
    h=lambda l: stats.norm.pdf(be,0,l)*2/(pi*(1+l*l))
    return integrate.quad(h,0,1)[0]+integrate.quad(h,1,np.inf)[0]
for be in [.1,1,3]: print(be,hs(be),K/2*log(1+4/be**2),K*log(1+2/be**2))
# posterior kappa
for y in [0,.5,1,2,3,5,10]:
    w=lambda k:(1-k)**-.5*exp(-k*y*y/2)
    Z=integrate.quad(w,0,1)[0]; m=integrate.quad(lambda k:k*w(k),0,1)[0]/Z
    print(y,round(m,3),round((1-m)*y,3))
print('tau.1',2/pi*atan(.1))
# beta aug
print(gamma(2.5)/gamma(12.5), integrate.quad(lambda e:e**1.5*(1-e)**9,0,1)[0]/gamma(10))
# exact Ewens posterior
def post(K,n=30,a0=1,b0=.2):
    th=np.linspace(1e-4,80,400001)
    lp=(K+a0-1)*np.log(th)-b0*th-(special.gammaln(th+n)-special.gammaln(th))
    w=np.exp(lp-lp.max()); w/=w.sum()
    m=(th*w).sum(); sd=sqrt(((th-m)**2*w).sum()); c=np.cumsum(w)
    q=[th[np.searchsorted(c,u)] for u in (.05,.5,.95)]
    return m,sd,q,th,w
for K in [3,8,15]:
    m,sd,q,_,_=post(K); print(K,round(m,3),round(sd,3),[round(x,3) for x in q])
# acceptance prob P(K=8) and widened tolerance
th=np.linspace(1e-4,80,40001); dth=th[1]-th[0]
prior=0.2*np.exp(-0.2*th)
n=30
# |s(30,k)| via polynomial
c=np.zeros(n+1); c[0]=1
for m in range(n):
    c=np.concatenate([[0],c[:-1]])+m*c  # multiply by (theta+m)
lgr=special.gammaln(th+n)-special.gammaln(th)
def pK(k): return np.exp(np.log(c[k])+k*np.log(th)-lgr)
print('acc',(pK(8)*prior).sum()*dth)
w=sum(pK(k) for k in range(6,11))*prior
print('acc wide',w.sum()*dth); w/=w.sum(); m=(th*w).sum(); print('sd wide',sqrt(((th-m)**2*w).sum()),m)
