# Worked one locus example: 3x3 (ordered 2x2) genotype table, non HW
import numpy as np
from fractions import Fraction as F
P11,Het,P22 = 0.30,0.30,0.40
P=np.array([[P11,Het/2],[Het/2,P22]])
m=np.array([[0.10,0.30],[0.30,-0.20]])
p=P.sum(1); print('p',p)
mbar=(P*m).sum(); print('mbar',mbar)
# normal equations: sum_v P_uv (m_uv - b_u - b_v)=0
# u=1: P11(m11-2b1)+x(m12-b1-b2)=0 ; u=2 similarly
x=Het/2
M=np.array([[2*P11+x, x],[x, 2*P22+x]])
rhs=np.array([P11*m[0,0]+x*m[0,1], P22*m[1,1]+x*m[1,0]])
print('M',M,'rhs',rhs)
b=np.linalg.solve(M,rhs); print('beta',b)
A=b[:,None]+b[None,:]; d=m-A; print('A',A,'delta',d)
print('normal check',(P*d).sum(1))
sigG=(P*(m-mbar)**2).sum(); sigA=(P*(A-mbar)**2).sum(); sigD=(P*d**2).sum()
print('sigG sigA sigD',sigG,sigA,sigD, sigA+sigD)
D=m[0,0]-2*m[0,1]+m[1,1]
J=1/P11+4/Het+1/P22
print('D',D,'J',J,'D^2/J',D*D/J, 'c=D/J',D/J, 'h',1/P11,-1/x/1, 1/P22)
print('delta from h: 11',D/J/P11,' 12',D/J*(-2/Het),' 22', D/J/P22)
alpha=np.log(P11*P22/(Het/2)**2); print('alpha',alpha, 'HW pq', p[0]*p[1])
# Fisher info of alpha at fixed p, numerically
def probs(pp,al):
    from scipy.optimize import brentq
    f=lambda xx: np.log((pp-xx)*(1-pp-xx)/xx**2)-al
    xx=brentq(f,1e-12,min(pp,1-pp)-1e-12)
    return np.array([pp-xx,2*xx,1-pp-xx])
e=1e-6
q0=probs(p[0],alpha); dq=(probs(p[0],alpha+e)-probs(p[0],alpha-e))/(2*e)
print('info alpha at fixed p', (dq**2/q0).sum(), '1/J',1/J)
# full info matrix (p,alpha) inverse
dqp=(probs(p[0]+e,alpha)-probs(p[0]-e,alpha))/(2*e)
Jm=np.array([[ (dqp**2/q0).sum(), (dqp*dq/q0).sum()],[(dqp*dq/q0).sum(),(dq**2/q0).sum()]])
print('full info',Jm,'inv',np.linalg.inv(Jm))
# velocity 1: pure selection Pdot = P (m - mbar)
Pd=P*(m-mbar)
pd=Pd.sum(1)
print('Pdot',Pd,'pdot',pd)
tang=(Pd*A).sum(); R=(Pd*d).sum()
print('tangential 2 sum pdot b',2*(pd*b).sum(),tang,'R',R,'total',(Pd*m).sum())
adot=Pd[0,0]/P11-2*(2*Pd[0,1])/Het+Pd[1,1]/P22
print('alphadot',adot,'D adot/J',D*adot/J)
# Ewens form 2 sum p bdot, finite difference
def beta_of(P):
    x=P[0,1]; M=np.array([[2*P[0,0]+x,x],[x,2*P[1,1]+x]]); rhs=np.array([P[0,0]*m[0,0]+x*m[0,1],P[1,1]*m[1,1]+x*m[1,0]])
    return np.linalg.solve(M,rhs)
h=1e-6
bd=(beta_of(P+h*Pd)-beta_of(P-h*Pd))/(2*h); print('bdot',bd,'2 sum p bdot',2*(p*bd).sum())
# Kimura form
Q=P/np.outer(p,p); Qd=(Pd*np.outer(p,p)-P*(np.outer(pd,p)+np.outer(p,pd)))/np.outer(p,p)**2
print('Kimura', (np.outer(p,p)*d*Qd).sum())
# velocity 2: tangential HW-type move with same pdot: Pdot_uv = pdot_u p_v + p_u pdot_v? not tangent unless P in HW; show
Pt=np.outer(pd,p)+np.outer(p,pd)
print('Pt',Pt,'R for product-rule velocity',(Pt*d).sum())
# velocity 3: inbreeding-type move: Pdot = eps*(+1, -1, -1 ... ) toward HW
Pr=np.array([[-1,0.5],[0.5,-1]])*0.0  # placeholder
# approach to HW under random mating in continuous time: Pdot = p p^T - P
Pm=np.outer(p,p)-P; print('mating velocity', Pm, 'pdot',Pm.sum(1),'R',(Pm*d).sum(), 'dmbar',(Pm*m).sum())
adm=Pm[0,0]/P11-2*(2*Pm[0,1])/Het+Pm[1,1]/P22; print('alphadot mating',adm, D*adm/J)
# HW comparison: same p in HW
PH=np.outer(p,p); bH=beta_of(PH); AH=bH[:,None]+bH[None,:]; dH=m-AH; mH=(PH*m).sum()
print('HW beta',bH,'sigA',(PH*(AH-mH)**2).sum(),'sigD',(PH*dH**2).sum(),'(pqD)^2',(p[0]*p[1]*D)**2)
