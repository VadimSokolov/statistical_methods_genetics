# Re-run the genome scaling experiment (sparsity 0.05, sd 3) with the posterior stated in the appendix.
import numpy as np, sys
from scipy.optimize import minimize_scalar
G=600
u=(np.arange(G)+0.5)/G; k=1-u*u; wq=2/G
def fit(y):
    def nll(lt):
        tau=np.exp(lt); W=np.exp(-np.outer(y*y,k)/2)/(1-(1-tau*tau)*k)*wq
        return -np.sum(np.log(W.sum(1)*tau/np.pi))
    r=minimize_scalar(nll,bounds=(np.log(0.003),np.log(3)),method='bounded',options={'xatol':1e-3})
    tau=np.exp(r.x); W=np.exp(-np.outer(y*y,k)/2)/(1-(1-tau*tau)*k)*wq; W/=W.sum(1,keepdims=True)
    mu=np.outer(y,np.ones(G))*(1-k); v=np.ones_like(mu)*(1-k)
    m2=mu*mu+v; m4=mu**4+6*mu*mu*v+3*v*v
    E2=(W*m2).sum(1); E4=(W*m4).sum(1); Ed=(W*mu).sum(1)
    return tau, E2.sum(), np.sqrt((E4-E2**2).sum()), (Ed**2).sum()
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 7)
for L,R in [(500,40),(2000,40),(5000,30),(20000,20)]:
    out=[]
    for rep in range(R):
        nz=rng.random(L)<0.05; d=np.where(nz,rng.normal(0,3,L),0.0); y=d+rng.normal(size=L)
        tau,pm,psd,plug=fit(y); t=(d*d).sum()
        out.append((t,pm,psd,plug,(y*y).sum()-L, abs(pm-t)<1.96*psd))
    o=np.array(out,dtype=float)
    print(L,'truth',o[:,0].mean().round(1),'postmean',o[:,1].mean().round(1),'rel bias',((o[:,1]-o[:,0])/o[:,0]).mean().round(3),
          'MC sd of (pm-truth)',(o[:,1]-o[:,0]).std().round(1),'rel',((o[:,1]-o[:,0])/o[:,0]).std().round(3),
          'post sd',o[:,2].mean().round(1),'plug rel bias',((o[:,3]-o[:,0])/o[:,0]).mean().round(3),
          'unshrunk rel sd',((o[:,4]-o[:,0])/o[:,0]).std().round(3),'coverage',o[:,5].mean().round(2),flush=True)
