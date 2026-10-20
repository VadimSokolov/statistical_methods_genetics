import numpy as np
# overdominance
wAA,wAS,wSS=0.9,1.0,0.2
q=0.05;p=1-q
wA=p*wAA+q*wAS; wS=p*wAS+q*wSS; wb=p*wA+q*wS
aA=wA-wb; aS=wS-wb
s2A=2*(p*aA**2+q*aS**2)
tot=p*p*wAA**2+2*p*q*wAS**2+q*q*wSS**2-wb**2
s2D=tot-s2A
qn=q*wS/wb; pn=1-qn; wbn=pn*pn*wAA+2*pn*qn*wAS+qn*qn*wSS
print('od',wb,aA,aS,s2A,s2D,s2D/s2A,(s2A+s2D)/s2A,s2A/wb,wbn-wb,(s2A/wb-(wbn-wb))/(wbn-wb))
q=1/9;p=1-q;wb=p*p*wAA+2*p*q*wAS+q*q*wSS;print('qstar var',p*p*wAA**2+2*p*q*wAS**2+q*q*wSS**2-wb**2)
# small example
G=np.array([[0,1,2,1,0],[1,0,1,2,1],[2,1,0,1,1],[1,2,1,0,2],[0,1,1,1,0],[1,1,2,0,1],[2,0,1,1,2],[1,2,0,2,1]],float)
y=np.array([-0.050,-0.015,0.036,-0.015,-0.073,-0.020,0.053,0.037])
print(G.mean(0),G.var(0))
X=G-1;A=X.T@X;print(A);b=X.T@y;print(b)
ah=np.linalg.solve(A,b);print(ah)
n=8;V=A/n;print(ah@V@ah,np.sum(np.diag(V)*ah**2))
Ai=np.linalg.inv(A);vif=np.diag(A)*np.diag(Ai);print(vif,vif.sum())
print(np.corrcoef(X.T).round(3))
Xc=X  # columns mean zero already
yc=y-y.mean()
# residual var with intercept
Z=np.column_stack([np.ones(8),X]);beta=np.linalg.lstsq(Z,y,rcond=None)[0];r=y-Z@beta;print('s2',r@r/2,beta)
print(ah@V@ah-5*(r@r/2)/8)
al=np.array([0.05,0,0,-0.03,0]);print(al@V@al,np.sum(np.diag(V)*al**2))
r0=y-X@ah; print('noint df3',r0@r0/3, ah@V@ah-5*(r0@r0/3)/8)
