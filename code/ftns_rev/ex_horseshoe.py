import numpy as np
from scipy import integrate, optimize
y=np.array([0.4,-1.2,0.9,2.1,-0.3,4.5,0.1,-2.7,1.6,-0.8,6.0,0.6])
# posterior of kappa given y, tau: p(k|y,tau) ∝ exp(-k y^2/2) (1-k)^{-1/2} / (1-(1-tau^2)k)
# derive from: lambda~C+, kappa=1/(1+tau^2 lambda^2); y|kappa ~ N(0,1/kappa): density sqrt(kappa) exp(-kappa y^2/2)
def post(yv,tau):
    # substitute k = 1-u^2 to remove (1-k)^{-1/2} singularity: dk = -2u du, (1-k)^{-1/2}=1/u -> integrand*2
    def f(u,pw): k=1-u*u; return 2*k**pw*np.exp(-k*yv*yv/2)/(1-(1-tau*tau)*k)
    Z=integrate.quad(lambda u: f(u,0),0,1,limit=200,epsabs=1e-14)[0]
    # prior normaliser: prior density of kappa ∝ k^{-1/2}(1-k)^{-1/2}/(1-(1-tau^2)k) ; marginal y|k ∝ k^{1/2}
    Ek=integrate.quad(lambda u: f(u,1),0,1,limit=200,epsabs=1e-14)[0]/Z
    Ek2=integrate.quad(lambda u: f(u,2),0,1,limit=200,epsabs=1e-14)[0]/Z
    return Z,Ek,Ek2
def prior_norm(tau):
    return integrate.quad(lambda u: 2/np.sqrt(1-u*u)/(1-(1-tau*tau)*(1-u*u)),0,1,limit=200)[0]
def marg_loglik(tau):
    c=prior_norm(tau)
    return sum(np.log(post(v,tau)[0]/c/np.sqrt(2*np.pi)) for v in y)
# sanity: prior norm should be pi/tau
print('prior norm', prior_norm(0.3), np.pi/0.3)
# check retained-fraction table entries
for tau in [0.5,0.18,0.05]:
    print(tau,[round(1-post(v,tau)[1],3) for v in [0,2,3,4,6,8]])
taus=np.exp(np.linspace(np.log(0.02),np.log(5),200)); ll=[marg_loglik(t) for t in taus]
i=int(np.argmax(ll)); res=optimize.minimize_scalar(lambda lt:-marg_loglik(np.exp(lt)),bracket=(np.log(taus[max(i-1,0)]),np.log(taus[i]),np.log(taus[min(i+1,199)])))
th=np.exp(res.x); print('tau hat',th, 'll',-res.fun, 'll at tau=1',marg_loglik(1.0))
for tau in [th,1.0]:
    tot=0; totsq=0
    print('tau',tau)
    for v in y:
        Z,Ek,Ek2=post(v,tau)
        Ed=(1-Ek)*v
        # Var(delta|y)= E[Var(delta|k,y)]+Var(E[delta|k,y]) ; delta|k,y ~ N((1-k)y, 1-k)
        Ed2=(1-Ek)+ v*v*(1-2*Ek+Ek2)
        tot+=Ed**2; totsq+=Ed2
        print(f'  y={v:5.1f}  Ek={Ek:.3f}  Edelta={Ed:6.3f}  Edelta2={Ed2:6.3f}')
    print('  sum Ed^2',tot,'post mean sum delta^2',totsq,'unshrunk sum y^2-L',(y*y).sum()-len(y),'sum y^2',(y*y).sum())
