import numpy as np
from scipy import integrate, stats, special
def tail(a):
    # k = exp(-t) region near 0 and k near 1 handled by splitting
    g=lambda k: 2*stats.norm.sf(10*np.sqrt(k/(1-k)))
    lb=special.betaln(a,a)
    # near 0: k=e^{-t}, dk = k dt
    f0=lambda t: np.exp(a*(-t) + (a-1)*np.log1p(-np.exp(-t)) - lb)*g(np.exp(-t))
    I0=integrate.quad(f0,np.log(2),np.inf,limit=1000,epsabs=1e-14)[0]
    f1=lambda k: np.exp((a-1)*np.log(k)+(a-1)*np.log1p(-k)-lb)*g(k)
    I1=integrate.quad(f1,0,0.5,limit=1000,epsabs=1e-14)[0] if False else 0
    # k in [1/2,1): g tiny (10*1 sd) -> negligible
    return I0
for a in [1,0.5,0.05]:
    print(a,tail(a))
