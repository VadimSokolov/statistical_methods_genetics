# Weighted paintbox check: GEM(theta) size biased layout, sample with f_alpha (G(r)=r^alpha);
# induced partition of n=4 should be ESF(theta/alpha). Ranked layout for comparison.
import numpy as np
rng=np.random.default_rng(20261017)
theta,alpha,n,reps,Kst=2.0,2.0,4,1_000_000,80
def types(lab):
    out=[]
    for row in lab:
        _,c=np.unique(row,return_counts=True)
        out.append(tuple(sorted(c,reverse=True)))
    return out
def run(ranked, chunk=200_000):
    from collections import Counter
    C=Counter(); K1=0
    for _ in range(reps//chunk):
        B=rng.beta(1,theta,size=(chunk,Kst))
        W=B*np.cumprod(np.hstack([np.ones((chunk,1)),1-B[:,:-1]]),axis=1)
        if ranked: W=-np.sort(-W,axis=1)
        R=1-np.cumsum(W,axis=1)            # tail sums R_k, k=1..Kst
        y=rng.random((chunk,n))**(1/alpha) # y=1-X, P(y<=r)=r^alpha
        # interval index = #{k: R_k >= y}  (k-th interval is R_k < y <= R_{k-1})
        lab=(R[:,None,:]>=y[:,:,None]).sum(axis=2)
        # encode partition type
        s=np.sort(lab,axis=1)
        d=np.diff(s,axis=1)!=0
        # pattern of equalities among sorted labels -> block sizes
        for row in d:
            pass
        sizes=[]
        for row in s:
            _,c=np.unique(row,return_counts=True); sizes.append(tuple(sorted(c,reverse=True)))
        C.update(sizes)
    return {k:v/reps for k,v in C.items()}
from math import prod
def rising(x,m):
    r=1.0
    for i in range(m): r*=x+i
    return r
def esf_type(b,t):
    mult={(4,):1,(3,1):4,(2,2):3,(2,1,1):6,(1,1,1,1):1}[b]
    k=len(b)
    return mult*t**(k-1)/rising(t+1,3)*prod(rising(1,x-1) for x in b)
for ranked in (False,True):
    P=run(ranked)
    print('ranked' if ranked else 'size biased')
    for b in [(4,),(3,1),(2,2),(2,1,1),(1,1,1,1)]:
        p=P.get(b,0); se=np.sqrt(p*(1-p)/reps)
        print(b, round(p,4), '+-',round(se,4),' ESF(1)',round(esf_type(b,theta/alpha),4),' ESF(2)',round(esf_type(b,theta),4))
