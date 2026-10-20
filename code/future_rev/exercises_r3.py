"""Checks for the exercises added to chapters/future.tex in review round 3 (birthday losses, conditional coverage).

Run: python3 -I code/future_rev/exercises_r3.py
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
N2=20000
for n in (100,1000,5000):
    ed=N2*(1-(1-1/N2)**n); pairs=n*(n-1)/2/N2
    print(n,"lost",round(n-ed,2),"pairs",pairs,"n^2/4N",n*n/(2*N2),"pairs-lost",round(pairs-(n-ed),3))
# exact expected loss vs binom(n,2)/2N - binom(n,3)/(2N)^2
for n in (1000,5000):
    print(n,"2nd order", n*(n-1)/2/N2 - n*(n-1)*(n-2)/6/N2**2)
z=1.645
cov=lambda x: norm.cdf((z-x/2)/np.sqrt(.5))-norm.cdf((-z-x/2)/np.sqrt(.5))
for x in (0,2,3): print("cov",x,cov(x))
fx=lambda x: norm.pdf(x,0,np.sqrt(2))
p=2*norm.sf(2/np.sqrt(2)); print("P|X|>2",p)
num=2*quad(lambda x: cov(x)*fx(x),2,np.inf)[0]; print("cov |X|>2",num/p)
# overall marginal coverage check
print("marg",quad(lambda x: cov(x)*fx(x),-np.inf,np.inf)[0], 2*norm.cdf(z)-1)
