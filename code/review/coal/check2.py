import numpy as np
from scipy import integrate, stats
k=np.arange(2,501); th=5
print(np.sum(1/(th+k-1))/th)
for shape,scale in [(2,4),(4,25)]:
    f=lambda t: sum(t/(t+i) for i in range(30))*stats.gamma.pdf(t,shape,scale=scale)
    print(shape,scale,integrate.quad(f,0,np.inf,limit=200)[0])
