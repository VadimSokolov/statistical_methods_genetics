import numpy as np
X=np.array([[1,1,0],[1,-1,1],[1,0,-1],[-1,1,1],[-1,-1,0],[-1,0,-1]],float)
b=np.array([1,.5,0]);print(X.T@X)
for l in [0,2,6]:
    H=X@np.linalg.solve(X.T@X+l*np.eye(3),X.T)
    print(l,np.trace(H).round(3),np.trace(H.T@H).round(3),(np.sum((H@X@b)**2)/6).round(3),(np.sum((H@X@b)**2)/6+np.trace(H.T@H)/6).round(3))
