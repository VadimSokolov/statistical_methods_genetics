import mpmath as mp
mp.mp.dps=30
Q=lambda f: mp.quad(f,[0,1,10,100,mp.inf])
R=lambda t: t/((t+1)**2*(t+2)**2*(t+3))
C=Q(R); print('C',C, mp.log(2)+0.75*mp.log(3)-1.5)
m1=Q(lambda t:t*R(t)); print('m1',m1,2.5-2.25*mp.log(3),'mean',m1/C)
m2=Q(lambda t:t*t*R(t)); print('m2',m2,6.75*mp.log(3)-4*mp.log(2)-4.5,'E2',m2/C,'sd',mp.sqrt(m2/C-(m1/C)**2))
F=lambda A: mp.quad(R,[0,A])/C
print('median',mp.findroot(lambda A:F(A)-0.5,1),'lo',mp.findroot(lambda A:F(A)-0.025,0.1),'hi',mp.findroot(lambda A:F(A)-0.975,7))
pr=Q(lambda t:t*R(t)/(t+4))/C; print('pred',pr,(mp.mpf(7)/6+mp.mpf(17)/9*mp.log(2)-2.25*mp.log(3))/C)
th=mp.findroot(lambda t:sum(t/(t+i) for i in range(4))-2,1); print('mle',th,th/(th+4))
# prior predictive
pri=lambda t:1/((t+1)*(t+2)*mp.log(2))
s=[6,11,6,1]
for k in range(1,5): print('K4=',k, s[k-1]*Q(lambda t:t**k/(t*(t+1)*(t+2)*(t+3))*pri(t)))
print('theta=1',[s[k-1]/mp.mpf(24) for k in range(1,5)])
# PY sigma 1/2
Rp=lambda t:(t+0.5)/((t+1)**2*(t+2)**2*(t+3)); Cp=Q(Rp); print('Cp',Cp,0.5*mp.log(2)+0.625*mp.log(3)-1)
print('PY mean',Q(lambda t:t*Rp(t))/Cp,'median',mp.findroot(lambda A:mp.quad(Rp,[0,A])/Cp-0.5,0.6),'pred',Q(lambda t:(t+1)/(t+4)*Rp(t))/Cp)
# Mellin table rows
def gam(m,a,b,c,d):
    r=[(a+i-1)/mp.mpf(c) for i in range(1,b+1)]
    g=[(-1)**(i-1)*(-r[i-1])**d*mp.rf(-r[i-1],m)/(mp.factorial(i-1)*mp.factorial(b-i)) for i in range(1,b+1)]
    return r,g
def Mel(s,m,a,b,c,d):
    r,g=gam(m,a,b,c,d); return mp.pi/(c*mp.sin(mp.pi*s))*sum(gi*ri**(s-1) for gi,ri in zip(g,r))
def Mq(s,m,a,b,c,d): return Q(lambda x:x**(s-1)*x**d*mp.rf(x,m)/mp.rf(c*x+a,b))
for args in [(0.4,0,1.1,5,2,0),(2.3,0,1.1,5,2,0),(-0.5,2,0.7,6,1,0),(-1.7,1,1,5,2,1),(-1.7,2,0.7,6,1,0)]:
    print('mellin',args,Mel(*args),Mq(*args) if args[0]>-1.0 or args[1]==1 else '')
