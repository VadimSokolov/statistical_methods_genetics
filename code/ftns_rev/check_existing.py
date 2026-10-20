import numpy as np
from math import *
# Jeffreys approx
for t in [2,3.34,4.72]:
    nu=100; print('K',t, sqrt(pi*nu/2)*(1+t*t/nu)**(-(nu-1)/2))
# Example A
w=np.array([[0.9,1.2],[1.2,0.7]]); p=np.array([0.3,0.7])
def stepA(w,p):
    wb=p@w@p; wi=w@p; a=wi-wb; dp=p*a/wb
    D=w[0,0]-2*w[0,1]+w[1,1]
    sA=2*np.sum(p*a**2); R=D*dp[0]**2
    wb2=(p+dp)@w@(p+dp)
    return wb,wi,a,dp,sA,R,wb2-wb
wb,wi,a,dp,sA,R,dw=stepA(w,p); print(wb,wi,a,dp,sA,R,sA/wb,dw, R/(sA/wb))
G=np.sum(np.outer(p,p)*(w-wb)**2); print('sigG',G,'sigD',G-sA)
for s in [0.3,0.1,0.03,0.01,0.003,0.001]:
    aa=(w-1)/0.3; ws=1+s*aa
    r=stepA(ws,p); print(s, r[4]/r[0], r[5], abs(r[5])/(r[4]/r[0]), abs(r[5])/(r[4]/r[0])/s)
# scale example
for c in [1.5,1.25]:
    ws=np.array([[c*c,c],[c,1]]); r=stepA(ws,p); print('scale',c, r[5]/(r[4]/r[0]))
# XOR example B
W=np.full((4,4),0.6)
# gametes AB,Ab,aB,ab ; genotype het both: AB/ab (0,3),(3,0),(1,2),(2,1); AAbb: (1,1); aaBB:(2,2)
for (i,j) in [(0,3),(3,0),(1,2),(2,1),(1,1),(2,2)]: W[i,j]=1.2
x=np.array([0.05,0.45,0.45,0.05]); r=0.5
for g in range(5):
    wb=x@W@x; D=x[0]*x[3]-x[1]*x[2]; Wx=W@x
    xn=np.array([x[0]*Wx[0]-r*W[0,3]*D, x[1]*Wx[1]+r*W[0,3]*D, x[2]*Wx[2]+r*W[0,3]*D, x[3]*Wx[3]-r*W[0,3]*D])/wb
    wb2=xn@W@xn
    print(g, x[0],x[1],wb,D,wb2-wb); x=xn
# ESF
th=5; print(sum(th/(th+i) for i in range(100)))
from scipy.optimize import brentq
print(brentq(lambda t: sum(t/(t+i) for i in range(100))-15, 0.1,50))
# Ockham
for L in [1000,10000]: print((L/(log(2)*0.67))**2)
print(0.67*sqrt(1e5))
