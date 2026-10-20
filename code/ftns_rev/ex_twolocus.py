# Two locus haploid example off linkage equilibrium
import numpy as np
x=np.array([0.40,0.10,0.20,0.30])   # AB Ab aB ab
m=np.array([0.20,0.00,0.10,0.40])
nA=np.array([1,1,0,0.]); nB=np.array([1,0,1,0.])
pA=x@nA; pB=x@nB; D=x[0]*x[3]-x[1]*x[2]
print('pA pB D',pA,pB,D, 'LE would be',pA*pB)
X=np.column_stack([np.ones(4),nA,nB])
Wt=np.diag(x)
b=np.linalg.solve(X.T@Wt@X, X.T@Wt@m); print('coef (const,bA,bB)',b)
print('normal matrix',X.T@Wt@X,'rhs',X.T@Wt@m)
A=X@b; d=m-A; mbar=x@m
print('mbar',mbar,'A',A,'delta',d)
sigG=x@(m-mbar)**2; sigA=x@(A-mbar)**2; sigI=x@d**2
E=m[0]-m[1]-m[2]+m[3]; J=(1/x).sum(); h=np.array([1,-1,-1,1])/x
print('sigG sigA sigI',sigG,sigA,sigI,'E',E,'J',J,'E^2/J',E*E/J,'c',E/J,'delta via h',E/J*h)
lam=np.log(x[0]*x[3]/(x[1]*x[2])); print('lambda',lam)
# average effects as allele substitution: beta_A=bA
# velocity 1: selection
Pd=x*(m-mbar); print('Pdot sel',Pd, 'pAdot',Pd@nA,'pBdot',Pd@nB)
print('tangential',Pd@A,'R',Pd@d,'lamdot',Pd@h,'E lamdot/J',E*(Pd@h)/J,'total',Pd@m)
print('tangential via 2 sum: bA pAdot + bB pBdot', b[1]*(Pd@nA)+b[2]*(Pd@nB))
# velocity 2: recombination, continuous time rate r
r=0.5; Pr=-r*D*np.array([1,-1,-1,1.]); print('Pdot rec',Pr,'R',Pr@d,'-rDE',-r*D*E,'lamdot',Pr@h,'total',Pr@m)
# combined
Pc=Pd+Pr; print('combined total',Pc@m,'tang',Pc@A,'R',Pc@d)
# decomposition of log velocity into V and Vperp
v=Pd/x; # = m - mbar
vN=(v@ (x*h))/(h@(x*h))*h; vT=v-vN; print('vN',vN,'vT',vT, 'check vT in span', np.linalg.lstsq(X,vT,rcond=None)[1])
# same p in LE
xL=np.array([pA*pB,pA*(1-pB),(1-pA)*pB,(1-pA)*(1-pB)])
bL=np.linalg.solve(X.T@np.diag(xL)@X, X.T@np.diag(xL)@m); AL=X@bL; mL=xL@m
print('LE: coef',bL,'sigA',xL@(AL-mL)**2,'sigI',xL@(m-AL)**2,'E^2/J',E*E/(1/xL).sum(), 'mbar',mL)
