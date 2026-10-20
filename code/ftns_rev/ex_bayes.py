import numpy as np
from scipy.special import betaln, gammaln
from scipy import integrate
# ---------- exact binomial Bayes factor for d=0 (as in Table bf)
def logK(s,n,G=1600):
    s11,s12,s22=s
    # H1: product of Beta functions B(s+1,n-s+1)
    l1=sum(betaln(si+1,n-si+1) for si in s)
    # H0: integrate over t11,t22 uniform, t12=(t11+t22)/2  on log-scale grid, windowed
    def ll(a,b):
        t=(a+b)/2
        return (s11*np.log(a)+(n-s11)*np.log1p(-a)+s22*np.log(b)+(n-s22)*np.log1p(-b)
                +s12*np.log(t)+(n-s12)*np.log1p(-t))
    # find mode roughly
    g=np.linspace(1e-6,1-1e-6,801); AA,BB=np.meshgrid(g,g,indexing='ij'); L=ll(AA,BB)
    i,j=np.unravel_index(np.argmax(L),L.shape); a0,b0=g[i],g[j]
    w=8/np.sqrt(n)+0.02
    ga=np.linspace(max(1e-9,a0-w),min(1-1e-9,a0+w),G); gb=np.linspace(max(1e-9,b0-w),min(1-1e-9,b0+w),G)
    AA,BB=np.meshgrid(ga,gb,indexing='ij'); L=ll(AA,BB); M=L.max()
    I=np.trapezoid(np.trapezoid(np.exp(L-M),gb,axis=1),ga)
    l0=M+np.log(I)
    return l0-l1
# check table: expected data
for th,lab in [((0.45,0.60,0.35),'d=.2'),((0.45,0.42,0.35),'d=.02'),((0.45,0.40,0.35),'d=0')]:
    for n in [25,100,1000]:
        s=[n*t for t in th]
        print(lab,n,np.exp(logK(s,n)))
# worked example: hypothetical viability assay
for s,n in [((331,372,302),500),((331,318,302),500)]:
    k=np.exp(logK(s,n)); print('assay',s,n,'K',k,'1/K',1/k)
    th=np.array(s)/n; Dh=th[0]-2*th[1]+th[2]; var=(th*(1-th)/n)@np.array([1,4,1]); print(' thetahat',th,'Dhat',Dh,'se',np.sqrt(var),'z',Dh/np.sqrt(var),'dhat',th[1]-(th[0]+th[2])/2)
    nu=3*n-3; t=Dh/np.sqrt(var); print(' Jeffreys approx', np.sqrt(np.pi*nu/2)*(1+t*t/nu)**(-(nu-1)/2))
# ---------- HWE exact BF on MN blood groups
n11,n12,n22=298,489,213; n=n11+n12+n22
p=(2*n11+n12)/(2*n); print('pM',p)
lm1=np.log(2)+gammaln(n11+1)+gammaln(n12+1)+gammaln(n22+1)-gammaln(n+3)
lm0=n12*np.log(2)+betaln(2*n11+n12+1,n12+2*n22+1)
print('K_HW MN',np.exp(lm0-lm1))
P=np.array([n11,n12,n22])/n; al=np.log(P[0]*P[2]/(P[1]/2)**2); J=1/P[0]+4/P[1]+1/P[2]
print('alpha hat',al,'J',J,'se',np.sqrt(J/n),'z',al/np.sqrt(J/n),'expected HW',n*np.array([p*p,2*p*(1-p),(1-p)**2]))
chi=((np.array([n11,n12,n22])-n*np.array([p*p,2*p*(1-p),(1-p)**2]))**2/(n*np.array([p*p,2*p*(1-p),(1-p)**2]))).sum(); print('chi2',chi)
# scarlet tiger moth
for c in [(1469,138,5)]:
    n11,n12,n22=c; n=sum(c); p=(2*n11+n12)/(2*n)
    lm1=np.log(2)+gammaln(n11+1)+gammaln(n12+1)+gammaln(n22+1)-gammaln(n+3)
    lm0=n12*np.log(2)+betaln(2*n11+n12+1,n12+2*n22+1)
    P=np.array(c)/n; al=np.log(P[0]*P[2]/(P[1]/2)**2); J=1/P[0]+4/P[1]+1/P[2]
    E=n*np.array([p*p,2*p*(1-p),(1-p)**2]); chi=((np.array(c)-E)**2/E).sum()
    print('moth p',p,'K',np.exp(lm0-lm1),'alpha',al,'se',np.sqrt(J/n),'E',E,'chi',chi)
