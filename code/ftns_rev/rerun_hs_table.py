# Re-run the genome simulation of Table hs with the posterior stated in the chapter appendix
import numpy as np
u=(np.arange(4000)+0.5)/4000; k=1-u*u; wq=2/4000   # k = 1-u^2 removes (1-k)^{-1/2}
def moments(y,tau):
    den=1-(1-tau*tau)*k
    W=np.exp(-np.outer(y*y,k)/2)/den*wq     # L x G
    Z=W.sum(1); Ek=(W*k).sum(1)/Z; Ek2=(W*k*k).sum(1)/Z
    return Z,Ek,Ek2
def loglik(y,tau):
    Z,_,_=moments(y,tau); c=np.pi/tau
    return np.sum(np.log(Z/c))
rng=np.random.default_rng(2024)
for frac,sd in [(0.10,3.0),(0.05,3.0),(0.02,2.5),(0.01,4.0)]:
    L=5000; nz=rng.random(L)<frac; d=np.where(nz,rng.normal(0,sd,L),0.0); y=d+rng.normal(size=L)
    taus=np.exp(np.linspace(np.log(0.005),np.log(2),120)); ll=[loglik(y,t) for t in taus]; th=taus[int(np.argmax(ll))]
    Z,Ek,Ek2=moments(y,th); Ed=(1-Ek)*y; Ed2=(1-Ek)+y*y*(1-2*Ek+Ek2)
    sq=np.sort(Ed**2)[::-1]
    print(frac,sd,'truth',round((d*d).sum()),'unshrunk',round((y*y).sum()-L),'hs',round(Ed2.sum()),'sumEd2',round((Ed**2).sum()),'tau',round(th,3),'meanEk',round(Ek.mean(),3),'top5',round(sq[:L//20].sum()/sq.sum(),3))
