import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import sys; sys.path.insert(0,str(_ROOT / "code" / "pig_rev"))
import numpy as np
from scipy import special as sp, optimize, integrate
from piglib import *
P=print
# Feller example
th=mle(50,10); P('mle50,10',th,VarK(th,50),info(th,50),info(th,50)**-0.5)
# beta identity
for t,n in [(2.5,10),(40.146,50),(0.5,5)]:
    lhs=np.exp(sp.gammaln(t)-sp.gammaln(t+n)); rhs=integrate.quad(lambda e:e**(t-1)*(1-e)**(n-1),0,1,epsabs=0,epsrel=1e-12)[0]/sp.gamma(n)
    P('beta',t,n,lhs,rhs)
# gibbs example quadrature n=50 K=10
s=summ(*grid_post(50,10,gamma_lp(.01,.01))); P('q50',s)
rng=np.random.default_rng(1); ch=gibbs(50,10,.01,.01,100000,rng,burn=1000); P('gibbs',ch.mean(),ch.std(),np.quantile(ch,[.025,.975]),iat(ch)[0])
# butterflies
a=np.array([118,74,44,24,29,22,20,19,20,15,12,14,6,12,6]); r=np.arange(1,16)
P('butter',a.sum(),(r*a).sum(),np.sum((-1)**(r+1)*a))
# BvM table
for K in [3,5,10,20,40,80,160]:
    t=mle(1000,K); J=VarK(t,1000); se=t/np.sqrt(J); s=summ(*grid_post(1000,K,gamma_lp(.01,.01)))
    P('bvm',K,round(t,3),round(J,1),round(se,3),[round(t-1.96*se,3),round(t+1.96*se,3)],[round(t*np.exp(-1.96/np.sqrt(J)),3),round(t*np.exp(1.96/np.sqrt(J)),3)],round(s['mean'],3),round(s['lo'],3),round(s['hi'],3),round(s['skew'],2))
# regimes
for n,K in [(200,19),(200,4),(30,28)]:
    s=summ(*grid_post(n,K,gamma_lp(.01,.01))); P('reg',n,K,mle(n,K),{k:round(float(v),3) for k,v in s.items()})
# pig prior
pp=pigprior_lp(10,3,-1)
ph,thg,w=grid_post(1,0,lambda x:0*x,extra=None)
# prior median: density theta^(K0-1)Gamma/Gamma
phi=np.linspace(-15,15,400001); thv=np.exp(phi); lw=pp(thv)+phi; w=np.exp(lw-lw.max()); c=np.cumsum(w); c/=c[-1]; P('prior median',np.interp(.5,c,thv))
for n,K in [(15609,240),(30,28)]:
    s=summ(*grid_post(n,K,pp,lo=-12,hi=20)); P('pigprior',n,K,{k:round(float(v),3) for k,v in s.items()})
# two variances
def EKf(t,n): return EK(t,n)
for n,k in [(200,25),(3540,178),(15609,240)]:
    t=mle(n,k); V=VarK(t,n); m=t*np.log(1+n/t); mp=np.log(1+n/t)-n/(n+t); VP=m/mp**2
    ft=optimize.brentq(lambda x:x*np.log(1+n/x)-k,1e-3,1e4)
    L=np.log(n/t)
    P('twovar',n,k,round(t,3),EK(t,n),round(V,2),round(V/k,3),round(np.sqrt(t*t/V),3),round(np.sqrt(VP),3),round(ft,3),'approx',round(np.sqrt(t/(L-1)),3),round(np.sqrt(t/L),3))
# n=200,k=25 priors
hc=lambda x:-0.5*np.log(x)-np.log1p(x)
for nm,lp in [('ga',gamma_lp(.01,.01)),('hc',hc)]:
    for n,k in [(200,25),(200,3),(30,28)]:
        s=summ(*grid_post(n,k,lp,lo=-15,hi=25)); P(nm,n,k,{kk:round(float(v),3) for kk,v in s.items()})
# three block sampler
def tb(n,k,nu,T,rng):
    th=1.0; out=np.empty(T)
    for t in range(T):
        le=log_beta_draw(th,n,rng); xi=rng.gamma(nu,1/(1+th)); th=rng.gamma(k+nu/2,1/(xi-le)); out[t]=th
    return out
ch=tb(200,25,1,200000,np.random.default_rng(2)); P('3block',np.median(ch),np.quantile(ch,[.025,.975]),iat(ch)[0])
# numerical study
P('EK5',EK(5,1000)); s=summ(*grid_post(1000,27,gamma_lp(.01,.01))); P('post27',s,mle(1000,27),mle(1000,27)/np.sqrt(VarK(mle(1000,27),1000)))
for a0,b0 in [(1e-3,1e-3),(1e-2,1e-2),(2,.05),(1,1)]:
    s=summ(*grid_post(15609,240,gamma_lp(a0,b0))); P('williams',a0,b0,s['mean'],s['sd'])
t=6; P('se theta6 n800 K~30', mle(800,30), mle(800,30)/np.sqrt(VarK(mle(800,30),800)))
# exercise 12
t=39.355; P('ex12',t*(sp.digamma(t+7080)-sp.digamma(t+3540)),t*np.log(2))
# U_m posterior of E
ph,thv,w=grid_post(3540,178,gamma_lp(.01,.01)); p=w/w.sum(); EU=thv*(sp.digamma(thv+3540+12069)-sp.digamma(thv+3540))
o=np.argsort(EU); c=np.cumsum(p[o]); P('EU',np.sum(p*EU),np.interp([.025,.975],c,EU[o]),summ(ph,thv,w)['mean'],summ(ph,thv,w)['sd'],mle(3540,178))
