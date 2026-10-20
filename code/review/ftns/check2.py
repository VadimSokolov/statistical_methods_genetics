import numpy as np
from scipy import integrate, optimize, special, stats
# s table
for s in [0.3,0.1,0.03,0.01,0.003,0.001]:
    a=np.array([-1/3,2/3,-1]); w=1+s*a; p=0.3;q=0.7
    wb=p*p*w[0]+2*p*q*w[1]+q*q*w[2]; w1=p*w[0]+q*w[1]; w2=p*w[1]+q*w[2]
    a1=w1-wb;a2=w2-wb; sA=2*(p*a1*a1+q*a2*a2); dp=p*a1/wb; D=w[0]-2*w[1]+w[2]; R=D*dp*dp
    p1=p+dp; wb1=p1*p1*w[0]+2*p1*(1-p1)*w[1]+(1-p1)**2*w[2]
    print(s,'sA/wb %.3g R %.3g rel %.4f%% total-check %.2e'%(sA/wb,R,100*abs(R)/(sA/wb),wb1-wb-(sA/wb+R)), 50*s**-4)
# Example B
x=np.array([0.05,0.45,0.45,0.05]); r=0.5
nA=np.array([1,1,0,0]); nB=np.array([1,0,1,0])
Wm=np.zeros((4,4))
for i in range(4):
    for j in range(4):
        a=nA[i]+nA[j]; b=nB[i]+nB[j]
        dis=(a==1 and b==1) or (a==2 and b==0) or (a==0 and b==2)
        Wm[i,j]=1.2 if dis else 0.6
for g in range(5):
    wb=x@Wm@x; D=x[0]*x[3]-x[1]*x[2]
    xn=x*(Wm@x); sgn=np.array([-1,1,1,-1]); xn=xn+sgn*r*Wm[0,3]*D; xn/=wb
    wbn=xn@Wm@xn
    print(g,np.round(x[:2],5),round(wb,6),round(D,5),round(wbn-wb,6))
    x=xn
# ESF
th=5; print('EK',sum(th/(th+i) for i in range(100)))
print('thetahat',optimize.brentq(lambda t: sum(t/(t+i) for i in range(100))-15,0.1,50))
from math import factorial as F
def esf(part,n,th):
    a=np.zeros(n+1,int)
    for j in part: a[j]+=1
    rf=np.prod([th+i for i in range(n)])
    return F(n)/rf*np.prod([th**a[j]/(j**a[j]*F(a[j])) for j in range(1,n+1)])
parts=[(6,1,1),(5,2,1),(4,3,1),(4,2,2),(3,3,2)]
for t in [0.5,2,10]:
    v=np.array([esf(p,8,t) for p in parts]); print(t,np.round(v/v.sum(),6))
# Beta tails P(|theta|>10), theta|k ~ N(0,(1-k)/k)
for a in [(1,1),(0.5,0.5),(0.05,0.05)]:
    f=lambda k: stats.beta.pdf(k,*a)*2*stats.norm.sf(10/np.sqrt((1-k)/k))
    print(a,integrate.quad(f,0,1,limit=500,points=[1e-6,1e-4,1e-2])[0])
print(stats.beta.cdf(0.1,0.5,1),stats.beta.sf(0.9,0.5,1),stats.beta.cdf(0.1,.5,.5))
