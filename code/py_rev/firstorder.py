# First order law example: sigma=0.5, theta=1, alpha=2. E[K_n] vs E[S_*] n^{sigma_*}
import numpy as np
from scipy.special import gammaln, gamma
rng=np.random.default_rng(11)
s,t,a=0.5,1.0,2.0
ss=s/(a*(1-s)+s); kap=a*ss/s
C=a**ss*s**((a-1)*ss)*gamma(1-ss)*gamma(1-s+ss)/gamma(1-s)
def ES(p): return np.exp(gammaln(t+1)+gammaln(t/s+p+1)-gammaln(t/s+1)-gammaln(t+p*s+1))
ESs=C*ES(kap)
print('sigma*',ss,'kappa',kap,'C',C,'E S*',ESs,'E S',ES(1))
Kst=200_000; k=np.arange(1,Kst+1); reps=1500
ns=np.array([1e2,1e3,1e4,1e5,1e6,1e7])
EK=np.zeros((reps,len(ns))); EKu=np.zeros((reps,len(ns)))
for r in range(reps):
    B=rng.beta(1-s,t+k*s); R=np.concatenate([[1.0],np.cumprod(1-B)])
    p=-np.diff(R**a)
    EK[r]=[np.sum(-np.expm1(n*np.log1p(-p))) for n in ns]
m=EK.mean(0); se=EK.std(0)/np.sqrt(reps)
for n,mm,e in zip(ns,m,se):
    print('n=%.0e  E K_n %.2f +- %.2f   E S* n^s* - 1/2 = %.2f   ratio %.4f'%(n,mm,e,ESs*n**ss-0.5,mm/(ESs*n**ss-0.5)))
sl=np.diff(np.log(m))/np.diff(np.log(ns)); print('local slopes',np.round(sl,4))
# Shakespeare profile fit
from scipy.optimize import minimize_scalar
A=np.array([14376,4343,2292,1463,1043,837,638,519,430,364])
def ratio(j,s): return np.exp(gammaln(j-s)-gammaln(j+1)-gammaln(1-s))
j=np.arange(2,11)
f=lambda s: np.sum((np.log(A[1:]/A[0])-np.log(ratio(j,s)))**2)
print('Shakespeare LS profile sigma',minimize_scalar(f,bounds=(0,0.95),method='bounded').x)
