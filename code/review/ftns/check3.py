import numpy as np
from scipy import integrate, optimize, stats
def post(y,tau):
    f=lambda u,pw: 2*(1-u*u)**pw*np.exp(-(1-u*u)*y*y/2)/(1-(1-tau*tau)*(1-u*u))
    Z=integrate.quad(lambda u:f(u,0),0,1,limit=200,epsabs=1e-14)[0]
    return Z,integrate.quad(lambda u:f(u,1),0,1,limit=200,epsabs=1e-14)[0]/Z,integrate.quad(lambda u:f(u,2),0,1,limit=200,epsabs=1e-14)[0]/Z
for tau in [0.16431165478931034,0.07247728772017273,0.027169267560781894,0.015605771676319024]:
    print('thr',tau,optimize.brentq(lambda y:(1-post(y,tau)[1])-0.5,0.01,20))
for tau in [0.5,0.18,0.05]:
    print(tau,[round(1-post(v,tau)[1],3) for v in [0,2,3,4,6,8]])
y=np.array([6.0,4.5,-2.7,2.1,1.6,-1.2,0.9,-0.8,0.6,0.4,-0.3,0.1])
def ll(tau): return sum(np.log(post(v,tau)[0]*tau/np.pi/np.sqrt(2*np.pi)) for v in y)
r=optimize.minimize_scalar(lambda lt:-ll(np.exp(lt)),bounds=(np.log(0.05),np.log(5)),method='bounded'); th=np.exp(r.x)
print('tauhat',th,ll(th),ll(0.5),ll(0.2))
for tau in [0.2,th]:
    T=P=0
    for v in y:
        Z,Ek,Ek2=post(v,tau); Ed=(1-Ek)*v; Ed2=(1-Ek)+v*v*(1-2*Ek+Ek2); T+=Ed2; P+=Ed*Ed
        print('  %5.1f %.3f %6.3f'%(v,Ek,Ed))
    print(tau,'total',T,'plug',P)
print('sum y2 -12',(y*y).sum()-12)
print('Ek tau=1 y=6',post(6,1)[1], 2/36)
