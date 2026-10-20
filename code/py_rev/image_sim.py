# Monte Carlo check of image_eppf for (sigma,theta,alpha)=(0.5,1,2), size biased layout
import numpy as np
from collections import Counter
rng=np.random.default_rng(7)
s,t,a,n=0.5,1.0,2.0,4
reps,chunk,Kst=400_000,20_000,3000
C=Counter()
k=np.arange(1,Kst+1)
for _ in range(reps//chunk):
    B=rng.beta(1-s,t+k*s,size=(chunk,Kst))
    R=np.cumprod(1-B,axis=1)
    y=rng.random((chunk,n))**(1/a)
    lab=np.empty((chunk,n),dtype=int)
    for i in range(chunk):
        lab[i]=np.searchsorted(-R[i],-y[i],side='right')  # #{k: R_k >= y}
    for row in lab:
        _,c=np.unique(row,return_counts=True); C[tuple(sorted(c,reverse=True))]+=1
for b in [(4,),(3,1),(2,2),(2,1,1),(1,1,1,1)]:
    p=C[b]/reps; print(b,round(p,4),'+-',round(np.sqrt(p*(1-p)/reps),4))
