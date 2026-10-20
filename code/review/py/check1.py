from fractions import Fraction as Fr
from math import factorial, gamma, lgamma, exp
import numpy as np
def partitions(n, m=None):
    if m is None: m = n
    if n == 0: yield []; return
    for f in range(min(n, m), 0, -1):
        for r in partitions(n - f, f): yield [f] + r
def rising(x, m):
    r = Fr(1)
    for i in range(m): r *= (x + i)
    return r
def shape(sh, s, t):
    n=sum(sh); a={}
    for x in sh: a[x]=a.get(x,0)+1
    N=Fr(factorial(n))
    for j,aj in a.items(): N/= factorial(j)**aj*factorial(aj)
    num=Fr(1)
    for i in range(1,len(sh)): num*= t+i*s
    for x in sh: num*=rising(1-s,x-1)
    return N*num/rising(t+1,n-1)
pars=[(Fr(0),Fr(1)),(Fr(0),Fr(2)),(Fr(1,2),Fr(0)),(Fr(1,2),Fr(1)),(Fr(1,4),Fr(2))]
for n in (3,4):
    for sh in partitions(n): print(n, sh, [str(shape(sh,*p)) for p in pars])
    print("sums",[str(sum(shape(sh,*p) for sh in partitions(n))) for p in pars])
# moments table
def ES(p,s,t): return exp(lgamma(t+1)+lgamma(t/s+p+1)-lgamma(t/s+1)-lgamma(t+p*s+1))
s,t=0.5,1.0
for a in (0.25,0.5,1,2,4):
    ss=s/(a*(1-s)+s); k=a*ss/s
    C=a**ss*s**((a-1)*ss)*gamma(1-ss)*gamma(1-s+ss)/gamma(1-s)
    m1=C*ES(k,s,t); m2=C**2*ES(2*k,s,t); sd=(m2-m1**2)**.5
    print("mom",a,round(ss,3),round(k,3),round(C,4),round(m1,3),round(sd,3),round(sd/m1,3))
for (s,t,a) in ((0.5,1,2),(0.4,1,2),(0.6,1,1.5),(0.5,1,1),(0.5,2,2)):
    ss=s/(a*(1-s)+s); k=a*ss/s
    C=a**ss*s**((a-1)*ss)*gamma(1-ss)*gamma(1-s+ss)/gamma(1-s)
    print("const",s,t,a,round(ss,4),"C",round(C,4),"Z*/Z^k",round(C*gamma(1-s)**k/gamma(1-ss),4))
# worked table ratio n=1e6, sigma .5, theta? ratio count = n^(ss-s)*C ?? test
for a in (0.25,0.5,1,2,4):
    s=0.5; ss=s/(a*(1-s)+s); k=a*ss/s
    C=a**ss*s**((a-1)*ss)*gamma(1-ss)*gamma(1-s+ss)/gamma(1-s)
    print("worked",a,round(ss,3),round(k,3),"n^d C",round(1e6**(ss-s)*C,4),"with E S ratio theta=1", round(1e6**(ss-s)*C*ES(k,s,1)/ES(1,s,1),4), "1e9", 1e9**(ss-s)*C)
# shakespeare
for sg in (0.396,0.456):
    print(sg,[round(exp(lgamma(j-sg)-lgamma(j+1)-lgamma(1-sg)),3) for j in range(1,11)])
A=[14376,4343,2292,1463,1043,837,638,519,430,364]; print([round(x/A[0],3) for x in A])
from scipy.optimize import minimize_scalar
f=lambda sg: sum((np.log(A[j-1]/A[0]) - (lgamma(j-sg)-lgamma(j+1)-lgamma(1-sg)))**2 for j in range(2,11))
print("ls fit", minimize_scalar(f,bounds=(0.01,0.99),method='bounded').x)
print("binom sd ex", (3/7*4/7/1087)**.5, 3/7/(2*1087))
