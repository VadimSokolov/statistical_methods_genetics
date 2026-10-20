import numpy as np
from scipy import integrate, special as sp
from mpmath import mp, quad, beta as mbeta, mpf, log, gamma, inf
mp.dps=30
def Em(a,b,n):
    t2=mpf(1)/n
    f=lambda u: t2*u/(1+t2*u)*u**(b-1)*(1+u)**(-a-b)/mbeta(a,b)
    return n*quad(f,[0,1,n,100*n,inf])
def EK(al,th,n):
    if al==0: return sum(th/(th+j) for j in range(n))
    return float(gamma(th+1)/(al*gamma(th+al))*gamma(th+al+n)/gamma(th+n)-th/al)
def lead(al,th,n):
    return float(gamma(th+1)/(al*gamma(th+al)))*n**al if al>0 else th*np.log(n)
def a1(a,b,n): return float(n*mbeta(a,b+n-1)/mbeta(a,b))
for (a,b) in [(.5,.5),(1,1),(.5,1),(1,3)]:
    al,th=1-a,a+b-1
    for n in [100,1000,10000,100000]:
        print((a,b),n,round(EK(al,th,n),2),round(lead(al,th,n),2),round(a1(a,b,n),2),round(float(Em(a,b,n)),2))
# spectrum n=20
n=20
th=2
print('EK20',EK(0,2,20),EK(.5,0,20))
from math import comb
for j in range(1,7):
    e=float(comb(n,j)*quad(lambda x:x**(j-1)*(1-x)**(n-j)*x**0*(1-x)**(th-1)/mbeta(1,th),[0,1]))
    h=float(comb(n,j)*quad(lambda x:x**(j-1)*(1-x)**(n-j)/(mp.pi*mp.sqrt(x*(1-x))),[0,1]))
    print(j,round(e,3),round(th/j,3),round(h,3))
s=sum(j*float(comb(n,j)*quad(lambda x:x**(j-1)*(1-x)**(n-j)/(mp.pi*mp.sqrt(x*(1-x))),[0,1])) for j in range(1,n+1));print('sum',s)
print(Em(.5,1,1000))
