import numpy as np
from scipy import integrate
def mom(z,tau):
    pr=lambda k: k**-0.5*(1-k)**-0.5/(1+(tau**2-1)*k)
    po=lambda k: k**0.5*np.exp(-k*z*z/2)*pr(k)
    f=lambda g: integrate.quad(g,0,1,limit=200,points=[1e-8])[0]
    Z=f(po); Ek=f(lambda k:k*po(k))/Z; Ek2=f(lambda k:k*k*po(k))/Z
    m=z*(1-Ek); Et2=z*z*(1-2*Ek+Ek2)+1-Ek
    return Ek,m,Et2-m*m,Et2
for tau in [1,0.1]:
    T1=T2=0
    for z in [0.5,1,2,3,5]:
        Ek,m,s2,Et2=mom(z,tau); T1+=m*m;T2+=Et2
        print(tau,z,round(Ek,3),round(m,3),round(m*m,3),round(s2,3),round(Et2,3))
    print(round(T1,2),round(T2,2))
