import numpy as np
from scipy import stats
s=0.5
for th in [0,1,10]:
    K=1.0;K1=1.0;out={}
    for n in range(1,10**6+1):
        if n in (100,1000,10**4,10**5,10**6): out[n]=(round(K1/K,4), round((K1/K-s)*n**s,3))
        newp=(th+s*K)/(th+n)
        K1=K1*(1-(1-s)/(th+n))+newp; K=K+newp
    print(th,out)
print(stats.ncx2.sf(stats.chi2.ppf(.95,1),1,7.85), 7.85/0.03**2)
# calib
rng=np.random.default_rng(1);th=rng.standard_normal(2_000_000);x=th+rng.standard_normal(th.size)
m=np.abs(x)>2;print(m.mean(), (np.abs(th[m])<1.645).mean())
for xx in [0,2,3]:
    d=stats.norm(xx/2,np.sqrt(.5));print(xx,d.cdf(1.645)-d.cdf(-1.645))
# birthday
M=20000
for n in [100,1000,5000]:
    i=np.arange(1,n);pno=np.prod(1-i/M) if n<M else 0
    lost=n-M*(1-(1-1/M)**n)
    print(n,n*(n-1)/2/M,pno,lost)
for n in range(150,180):
    i=np.arange(1,n)
    if 1-np.prod(1-i/M)>=0.5: print('first n with P>=.5',n); break
print(np.sqrt(M*2*np.log(2)))
