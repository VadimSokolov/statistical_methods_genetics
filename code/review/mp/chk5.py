import numpy as np
from scipy.optimize import brentq
d=np.array([6,4,4.]);b=np.array([1,.5,0])
E=lambda l: (np.sum(d*(d/(d+l))**2*b**2)+np.sum((d/(d+l))**2))/6
print(brentq(lambda l:E(l)-7/6,0,10))
th=5
for n in [100,10**4,10**6]:
    j=np.arange(n);I=np.sum(j/(th+j)**2)/th;print(n,1/np.sqrt(I))
