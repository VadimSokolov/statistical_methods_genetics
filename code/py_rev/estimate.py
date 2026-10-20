# Estimate sigma from a simulated sample drawn with effort profile f_alpha (G(r)=r^alpha)
import numpy as np
rng=np.random.default_rng(314)
s,t,a,n,Kst,reps=0.6,5.0,2.0,1_000_000,1_000_000,100
ss=s/(a*(1-s)+s)
k=np.arange(1,Kst+1)
res=[]
for r in range(reps):
    B=rng.beta(1-s,t+k*s)
    R=np.concatenate([[1.0],np.cumprod(1-B)])
    out=[]
    for G in (lambda x:x**a,):
        p=-np.diff(G(R)); p=np.append(p,G(R[-1]))
        c=rng.multinomial(n,p/p.sum())
        A1=(c==1).sum(); A2=(c==2).sum(); K=(c>0).sum()
        out+= [K,A1,A2]
    res.append(out)
res=np.array(res,float)
def summ(K,A1,A2,alpha):
    e1=A1/K; e2=1-2*A2/A1; e3=np.log(K)/np.log(n)
    corr=lambda e: alpha*e/(alpha*e+1-e)
    return e1,e2,e3,corr(e1),corr(e2),corr(e3)
print('sigma*=',ss)
for lab,(K,A1,A2),al in [('weighted',res[:,0:3].T,a)]:
    E=summ(K,A1,A2,al)
    print(lab,' mean K %.1f A1 %.1f A2 %.1f'%(K.mean(),A1.mean(),A2.mean()))
    for name,e in zip(['A1/K','1-2A2/A1','logK/logn','corr A1/K','corr 1-2A2/A1','corr logK/logn'],E):
        print('  %-16s mean %.4f sd %.4f'%(name,e.mean(),e.std()))
    print('  first replicate K,A1,A2:',K[0],A1[0],A2[0])
# rate bias check: sigma*(1+1/(2K))
K,A1=res[:,0],res[:,1]
print('mean A1/K - sigma*(1+1/2K):',(A1/K-ss*(1+1/(2*K))).mean())

K0,A10,A20=res[0]
e1=A10/K0; e2=1-2*A20/A10
c=lambda e: a*e/(a*e+1-e)
print('first replicate: A1/K %.4f -> %.4f ; 1-2A2/A1 %.4f -> %.4f; logK/logn %.4f -> %.4f'%(e1,c(e1),e2,c(e2),np.log(K0)/np.log(n),c(np.log(K0)/np.log(n))))
