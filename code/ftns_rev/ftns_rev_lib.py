"""Horseshoe posterior for y ~ N(delta,1), delta ~ N(0, tau^2 lambda^2), lambda ~ C+(0,1),
computed on a grid in kappa = 1/(1+tau^2 lambda^2) using the exact conditional
p(kappa | y, tau) ∝ exp(-kappa y^2/2) (1-kappa)^{-1/2} / (1-(1-tau^2) kappa)  (chapter appendix).
tau is fitted by marginal maximum likelihood."""
import numpy as np
from scipy.optimize import minimize_scalar
G=600
u=(np.arange(G)+0.5)/G; k=1-u*u; wq=2/G          # k=1-u^2 removes the (1-k)^{-1/2} pole
def weights(y,tau):
    return np.exp(-np.outer(y*y,k)/2)/(1-(1-tau*tau)*k)*wq
def fit_tau(y):
    nll=lambda lt: -np.sum(np.log(weights(y,np.exp(lt)).sum(1)*np.exp(lt)/np.pi))
    r=minimize_scalar(nll,bounds=(np.log(0.003),np.log(3)),method='bounded',options={'xatol':1e-3})
    return np.exp(r.x)
def posterior(y,tau):
    W=weights(y,tau); W/=W.sum(1,keepdims=True)
    mu=np.outer(y,1-k); v=np.broadcast_to(1-k,mu.shape)
    m2=mu*mu+v; m4=mu**4+6*mu*mu*v+3*v*v
    return dict(Ek=(W*k).sum(1), Ed=(W*mu).sum(1), E2=(W*m2).sum(1), E4=(W*m4).sum(1))
def fit(y):
    tau=fit_tau(y); P=posterior(y,tau); P['tau']=tau
    P['total']=P['E2'].sum(); P['total_sd']=np.sqrt((P['E4']-P['E2']**2).sum()); P['plug']=(P['Ed']**2).sum()
    return P
def simulate(rng,L,frac,sd):
    nz=rng.random(L)<frac; d=np.where(nz,rng.normal(0,sd,L),0.0); return d, d+rng.normal(size=L)
