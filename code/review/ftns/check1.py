import numpy as np
from scipy import integrate, optimize, special, stats
# Example ls
P=np.array([[0.30,0.15],[0.15,0.40]]); m=np.array([[0.10,0.30],[0.30,-0.20]])
def ls(P,m):
    k=P.shape[0]; X=[]
    A=np.zeros((k,k)); b=np.zeros(k)
    for u in range(k):
        for v in range(k):
            e=np.zeros(k); e[u]+=1; e[v]+=1
            A+=P[u,v]*np.outer(e,e); b+=P[u,v]*m[u,v]*e
    beta=np.linalg.lstsq(A,b,rcond=None)[0]
    Af=beta[:,None]+beta[None,:]; return beta,Af,m-Af
beta,Af,d=ls(P,m); mb=(P*m).sum()
print('beta',beta,'delta',d,'sA',(P*(Af-mb)**2).sum(),'sD',(P*d*d).sum())
p=0.45;q=0.55;H=np.array([[p*p,p*q],[p*q,q*q]])
b2,A2,d2=ls(H,m); mb2=(H*m).sum(); print('HW beta',b2,'sA',(H*(A2-mb2)**2).sum(),'sD',(H*d2*d2).sum())
# velocity 1: selection; numeric dbeta
Pd=P*(m-mb); h=1e-7
bp,_,_=ls(P+h*Pd,m); bm,_,_=ls(P-h*Pd,m); db=(bp-bm)/(2*h); print('dbeta',db,'Ewens',2*(0.45*db[0]+0.55*db[1]))
# Kimura form
pp=P.sum(1); pd=Pd.sum(1)
r=lambda P: P/np.outer(P.sum(1),P.sum(1))
dr=(r(P+h*Pd)-r(P-h*Pd))/(2*h); print('Kimura',(np.outer(pp,pp)*d*dr).sum())
# MN
n11,n12,n22=298,489,213; n=1000
pM=(2*n11+n12)/2000; qM=1-pM
E=np.array([n*pM**2,2*n*pM*qM,n*qM**2]); O=np.array([n11,n12,n22])
print('pearson',((O-E)**2/E).sum())
lK=n12*np.log(2)+special.betaln(2*n11+n12+1,n12+2*n22+1)+special.gammaln(n+3)-np.log(2)-special.gammaln(n11+1)-special.gammaln(n12+1)-special.gammaln(n22+1)
print('K MN',np.exp(lK))
def KHW(a,b,c):
    nn=a+b+c
    return np.exp(b*np.log(2)+special.betaln(2*a+b+1,b+2*c+1)+special.gammaln(nn+3)-np.log(2)-special.gammaln(a+1)-special.gammaln(b+1)-special.gammaln(c+1))
print('K ex9',KHW(25,50,25),KHW(35,30,35))
# verify K formula directly: H1 Dirichlet(1,1,1) marginal = n!/(n11!n12!n22!) * 2!/(n+2)! * prod n_i! ... check via integration small case
a,b_,c=3,2,4; nn=9
from math import factorial as F
mult=F(nn)/(F(a)*F(b_)*F(c))
m1=mult*2/ (F(nn+2)/F(a)/F(b_)/F(c)) # Dirichlet(1,1,1): mult * Gamma(3)/Gamma(n+3) * prod Gamma(n_i+1)
m1=mult*2*F(a)*F(b_)*F(c)/F(nn+2)
m0=mult*integrate.quad(lambda p:(p*p)**a*(2*p*(1-p))**b_*((1-p)**2)**c,0,1)[0]
print('K check small',m0/m1,KHW(a,b_,c))
# two locus
x=np.array([0.4,0.1,0.2,0.3]); mm=np.array([0.2,0,0.1,0.4]); nA=np.array([1,1,0,0]); nB=np.array([1,0,1,0])
X=np.c_[np.ones(4),nA,nB]; W=np.diag(x)
bb=np.linalg.solve(X.T@W@X,X.T@W@mm); A=X@bb; dl=mm-A; mbar=x@mm
print('b',bb,'A',A,'delta',dl,'sG',x@(mm-mbar)**2,'sA',x@(A-mbar)**2,'sAB',x@dl**2)
xl=np.array([0.5*0.6,0.5*0.4,0.5*0.6,0.5*0.4])
bl=np.linalg.solve(X.T@np.diag(xl)@X,X.T@np.diag(xl)@mm); Al=X@bl; ml=xl@mm
print('LE b',bl,'sA',xl@(Al-ml)**2,'sAB',xl@(mm-Al)**2)
xd=x*(mm-mbar); print('xdot',xd,'R',xd@dl)
