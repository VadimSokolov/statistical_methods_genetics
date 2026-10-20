# Exact n<=4 partition probabilities of the weighted image of PD(sigma,theta) under G(r)=r^alpha,
# size biased layout: a RAM with residual factors Y_k = X_k^alpha, X_k ~ Beta(theta+k sigma, 1-sigma).
import numpy as np
from scipy.special import gammaln
from scipy.optimize import brentq, fsolve
def image_eppf(s,t,a,K=200000):
    k=np.arange(1,K+1); A=t+k*s; B=1-s
    def EY(m): # E[Y^m] = E[X^(a m)]
        return np.exp(gammaln(A+a*m)+gammaln(A+B)-gammaln(A)-gammaln(A+B+a*m))
    EYm={m:EY(m) for m in range(0,5)}
    def E1mY(j,i=0): # E[(1-Y)^j Y^i]
        from math import comb
        return sum(comb(j,r)*(-1)**r*EYm[r+i] for r in range(j+1))
    c={(j,i):E1mY(j,i) for j in range(5) for i in range(5) if i+j<=4}
    U={j:0.0 for j in (2,3,4)}; T1=0.0; T2=0.0
    for idx in range(K-1,-1,-1):
        for j in (2,3,4):
            U[j]=c[(j,0)][idx]+EYm[j][idx]*U[j]
        T2=c[(4,0)][idx]+2*c[(2,2)][idx]*T1+EYm[4][idx]*T2
        T1=c[(2,0)][idx]+EYm[2][idx]*T1
    m2,m3,m4,Q=U[2],U[3],U[4],T2
    p={(4,):m4,(3,1):m3-m4,(2,2):Q-m4,(2,1,1):m2-2*m3-Q+2*m4}
    p[(1,1,1,1)]=1-(p[(4,)]+4*p[(3,1)]+3*p[(2,2)]+6*p[(2,1,1)])
    return m2,m3,p
def ep4(s,t):
    from math import prod
    def rising(x,m):
        r=1.0
        for i in range(m): r*=x+i
        return r
    out={}
    for b in [(4,),(3,1),(2,2),(2,1,1),(1,1,1,1)]:
        kk=len(b)
        out[b]=prod([t+i*s for i in range(1,kk)])/rising(t+1,3)*prod(rising(1-s,x-1) for x in b)
    return out
mult={(4,):1,(3,1):4,(2,2):3,(2,1,1):6,(1,1,1,1):1}
def fitEP(m2,m3):
    # (1-s)/(t+1)=m2, (1-s)(2-s)/((t+1)(t+2))=m3
    def eq(v):
        s,t=v; return [(1-s)/(t+1)-m2,(1-s)*(2-s)/((t+1)*(t+2))-m3]
    return fsolve(eq,[0.3,1.0])
if __name__=='__main__':
    print('checks')
    m2,m3,p=image_eppf(0.5,1.0,1.0); print('alpha=1 vs PD(.5,1):',{b:round(p[b],6) for b in p},{b:round(v,6) for b,v in ep4(.5,1).items()})
    m2,m3,p=image_eppf(0.0,2.0,2.0,K=3000); print('sigma=0, theta=2, alpha=2 vs ESF(1):',{b:round(p[b],6) for b in p},{b:round(v,6) for b,v in ep4(0,1).items()})
    for (s,t,a) in [(0.5,1.0,2.0),(0.5,1.0,0.5),(0.25,2.0,2.0),(0.5,1,4.0)]:
        m2,m3,p=image_eppf(s,t,a)
        sf,tf=fitEP(m2,m3); e=ep4(sf,tf)
        print(f'\n(s,t,a)=({s},{t},{a}) m2={m2:.5f} m3={m3:.5f}  fitted EP sigma={sf:.4f} theta={tf:.4f}')
        for b in p:
            print(b, 'image type prob %.5f'%(mult[b]*p[b]),' EP fit %.5f'%(mult[b]*e[b]), ' ratio %.4f'%(p[b]/e[b]))
