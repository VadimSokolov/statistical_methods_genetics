import mpmath as mp, numpy as np, math
from scipy.special import exp1
print('ex8', math.exp(.01)*exp1(.01)-math.exp(.02)*exp1(.02))
print('mode', np.roots([4,15,9,-6]))
t=mp.mpf('0.3887'); 
import sympy as s
th=s.symbols('t'); e=s.expand((th+1)*(th+2)*(th+3)-2*th*(th+2)*(th+3)-2*th*(th+1)*(th+3)-th*(th+1)*(th+2)); print(e)
# k=1 kappa threshold
mp.mp.dps=60
for n in range(48,54):
    J=mp.quad(lambda x: mp.gamma(x+1)*mp.gamma(n)/mp.gamma(x+n),[0,0.1,1,10,mp.inf])
    print(n, float(mp.log10(math.comb(n-2,(n-2)//2)*(n-1)/J)))
# digits example n=40 double
n=40
r=[0.5,1.5]+list(range(1,n))
def coeffs(rr,num,prec):
    out=[]
    for i,ri in enumerate(rr):
        v=num(-ri)
        for l,rl in enumerate(rr):
            if l!=i: v=v/(rl-ri)
        out.append(v)
    return out
g=coeffs([float(x) for x in r],lambda a:a*a,None)
print('sum gamma double',sum(g),'max',max(abs(x) for x in g),'C double',-sum(gi*math.log(ri) for gi,ri in zip(g,r)))
mp.mp.dps=120
gm=coeffs([mp.mpf(x) for x in r],lambda a:a*a,None)
print('C mp',-sum(gi*mp.log(ri) for gi,ri in zip(gm,[mp.mpf(x) for x in r])))
