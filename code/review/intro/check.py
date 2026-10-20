import numpy as np
from scipy import stats, optimize
from fractions import Fraction as Fr
from math import comb, factorial
# Mendel
o=np.array([5474,1850]);e=7324*np.array([.75,.25]);X=((o-e)**2/e).sum();print('mendel',e,X,stats.chi2.sf(X,1),np.sqrt(7324*3/16))
# HW test
o=np.array([180,220,100]);p=(360+220)/1000;e=500*np.array([p*p,2*p*(1-p),(1-p)**2]);X=((o-e)**2/e).sum()
F=1-(220/500)/(2*p*(1-p));print('hw',e,X,stats.chi2.sf(X,1),F,500*F*F)
# Fisher
a,d,p=1,.5,.3;q=1-p;f=np.array([p*p,2*p*q,q*q]);g=np.array([a,d,-a]);x=np.array([2,1,0])
m=f@g;VG=f@(g-m)**2;mx=f@x;al=(f@((x-mx)*(g-m)))/(f@(x-mx)**2);bv=m+al*(x-mx)
print('fisher',m,VG,al,bv,g-bv,2*p*q*al**2,(2*p*q*d)**2)
VA=2*p*q*al**2;VD=(2*p*q*d)**2;VP=VG+1;print(VA/VP,VA/2/VP,(VA/2+VD/4)/VP)
# fixation
N=500
for s in [0.05,0.01,0.001,0.0001,0]:
    br=optimize.brentq(lambda x:1-x-np.exp(-(1+s)*x),1e-12,1) if s>0 else 0
    ku=(1-np.exp(-2*s))/(1-np.exp(-4*N*s)) if s>0 else 1/(2*N)
    M=2*N;i=np.arange(M+1);pp=i/M;pr=pp*(1+s+s*pp)/(1+2*s*pp)
    P=stats.binom.pmf(i[None,:],M,pr[:,None])
    A=np.eye(M-1)-P[1:M,1:M];b=P[1:M,M];u=np.linalg.solve(A,b)
    print(s,4*N*s,2*s,round(br,5),round(ku,5),round(u[0],5))
# ESF
def esf(c,th):
    n=sum(c);r=Fr(factorial(n));den=1
    for i in range(n): den*= (th+i)
    r/=den
    from collections import Counter
    for j,aj in Counter(c).items(): r*=Fr(th,j)**aj/factorial(aj)
    return r
cs=[[5],[4,1],[3,2],[3,1,1],[2,2,1],[2,1,1,1],[1]*5]
for th in [1,2]: print([str(esf(c,th)) for c in cs],sum(esf(c,th) for c in cs))
print(sum(Fr(1,1+i) for i in range(5)),sum(Fr(2,2+i) for i in range(5)))
# coal
ET=[2/(k*(k-1)) for k in range(2,6)];print(sum(ET),np.sqrt(sum(t*t for t in ET)),1/1.6)
a5=sum(1/j for j in range(1,5));print(a5,2*a5,5*a5+25*sum(1/j**2 for j in range(1,5)))
# exercises
o=np.array([6022,2001]);e=8023*np.array([.75,.25]);print(e,((o-e)**2/e).sum())
o=np.array([315,101,108,32]);e=556*np.array([9,3,3,1])/16;X=((o-e)**2/e).sum();print(X,stats.chi2.sf(X,3))
