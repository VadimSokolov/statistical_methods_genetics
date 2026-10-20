import numpy as np
from fractions import Fraction as F
from math import comb, log, factorial
from itertools import product
def a(n,r): return sum(1/j**r for j in range(1,n))
# n=4 table
q=[F(1,2),F(1,3),F(1,4)]
def h(s):
    tot=F(0)
    for i in range(s+1):
        for j in range(s+1-i):
            tot+=q[0]**i*q[1]**j*q[2]**(s-i-j)
    return tot
print([str(F(1,4)*h(s)) for s in range(8)])
# info table
def info(n,th):
    k=np.arange(2,n+1)
    IK=np.sum((k-1)/(th+k-1)**2)/th; IS=np.sum(1/(th+k-1))/th
    a1,a2=a(n,1),a(n,2); W=a1**2/(th*a1+th**2*a2)
    # exact I_S by numerical diff of pmf via convolution
    def pmf(t,smax):
        P=np.zeros(smax); P[0]=1
        for kk in range(2,n+1):
            p=(kk-1)/(kk-1+t); qq=1-p
            Q=np.zeros(smax)
            for s in range(smax):
                Q[s]=p*P[s]+(qq*Q[s-1] if s>0 else 0)
            P=Q
        return P
    smax=int(th*a1+40*np.sqrt(th*a1+th**2*a2))+50
    e=1e-5; P=pmf(th,smax); d=(np.log(pmf(th+e,smax))-np.log(pmf(th-e,smax)))/(2*e)
    Iex=np.sum(P*d**2)
    return IK,W,Iex,IS,log(n)/th
for n,th in [(4,1),(4,5),(30,1),(30,5),(500,1),(500,5)]:
    print(n,th,np.round(info(n,th),3))
# n=30 theta=10 mean var
n=30;print(10*a(30,1),10*a(30,1)+100*a(30,2),a(30,1),a(30,2))
# coupling conditional means n=4 theta=1
p={2:F(1,2),3:F(2,3),4:F(3,4)}
num={};den={}
for marks in product([0,1],repeat=3):
    pr=F(1); m=0
    for kk,mk in zip([2,3,4],marks):
        pr*= (1-p[kk]) if mk else p[kk]
        if mk: m+= 1/p[kk]
    K=1+sum(marks); num[K]=num.get(K,0)+pr*m; den[K]=den.get(K,0)+pr
print({K:(str(num[K]/den[K]),str(den[K])) for K in den})
# n3 example
for th in [0.5,2,8]: print(th,(2+th)/(2*(3+2*th)))
# Ewens E[H]
for th in [8,100]: print('EH',th,sum(th/(th+i) for i in range(30)), th*a(30,1))
# watterson example
a1,a2=a(11,1),a(11,2); S=18
print(a1,a2,S/a1, S/a1**2+a2/a1**2*S*(S-1)/(a1**2+a2))
v=S/a1**2+a2/a1**2*S*(S-1)/(a1**2+a2); print(np.sqrt(v), S/a1-1.645*np.sqrt(v),S/a1+1.645*np.sqrt(v))
print('var thetaW at 4', 4/a1+16*a2/a1**2)
# n=4 exact mean/var
print(F(11,6)+F(49,36), 11/6+49/36)
from scipy.special import exp1
print('ex11', np.e*exp1(1))
print('500', 8*a(500,1), 8*a(500,1)+64*a(500,2))
