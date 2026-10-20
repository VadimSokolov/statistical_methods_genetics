import numpy as np
from scipy import stats, special, integrate, optimize
from math import *
# fixation table
def wf_fix(N,s):
    psi=lambda x: x*(1+s)/(1+s*x)
    P=np.array([[stats.binom.pmf(j,N,psi(i/N)) for j in range(N+1)] for i in range(N+1)])
    Q=P[1:N,1:N]; R=P[1:N,N]
    u=np.linalg.solve(np.eye(N-1)-Q,R); return u[0]
def kim(N,s,x): return (1-exp(-2*N*s*x))/(1-exp(-2*N*s))
for N in [10,20,50]:
    for s in [.01,.05,.1]:
        e=wf_fix(N,s); k=kim(N,s,1/N); r=1+s; m=(1-1/r)/(1-r**-N); kh=kim(N/2,s,1/N)
        print(N,s,round(e,4),round(k,4),f"{100*abs(k-e)/e:.1f}%",round(m,4), f"half {100*abs(kh-m)/m:.1f}%")
# absorption time
for N in [10,20,50]:
    P=np.array([[stats.binom.pmf(j,N,i/N) for j in range(N+1)] for i in range(N+1)])
    Q=P[1:N,1:N]; t=np.linalg.solve(np.eye(N-1)-Q,np.ones(N-1)); print('T',N,round(t[N//2-1],1),round(2*N*log(2),1))
# stick
def tail(a,t,k): return np.prod([(t+i*a)/(t+1+(i-1)*a) for i in range(1,k+1)])
print('tail0.5,25',tail(.5,5,25))
k=1
while tail(.5,5,k)>=.01: k+=1
print('k',k)
k=1
while tail(0,5,k)>=.01: k+=1
print('k0',k)
# Williams
N=15609;S=240
aF=optimize.brentq(lambda a:a*log(1+N/a)-S,1,200); print('aF',aF,N/(N+aF))
f=lambda t:t*(special.digamma(t+N)-special.digamma(t))-S
th=optimize.brentq(f,1,200)
I=(sum(th*m/(th+m)**2 for m in range(N)))/th**2
print('th',th,1/sqrt(I))
def EA(j,t,n): return t/j*np.prod([(n-i)/(t+n-1-i) for i in range(j)])
x=N/(N+aF)
print([round(EA(j,th,N),2) for j in [1,2,3,10]],[round(aF*x**j/j,2) for j in [1,2,3,10]], th*log(N), N/th)
