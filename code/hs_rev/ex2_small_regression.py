# L = 5 loci, n = 8 individuals: normal equations, plug in, exact bias
import numpy as np
rng = np.random.default_rng(7)
G = np.array([[0,1,2,1,0],
              [1,0,1,2,1],
              [2,1,0,1,1],
              [1,2,1,0,2],
              [0,1,1,1,0],
              [1,1,2,0,1],
              [2,0,1,1,2],
              [1,2,0,2,1]], float)
n, L = G.shape
alpha = np.array([0.05, 0.0, 0.0, -0.03, 0.0])
sig = 0.05
eps = np.round(rng.standard_normal(n)*sig, 3)
X = G - G.mean(0)
y = np.round(X @ alpha + eps, 3)   # recorded fitness deviations, 3 decimals
print("G\n", G.astype(int)); print("colmeans", G.mean(0))
print("y", y, "eps", eps)
XtX = X.T @ X; Xty = X.T @ y
print("XtX\n", np.round(XtX, 3)); print("Xty", np.round(Xty, 4))
ahat = np.linalg.solve(XtX, Xty)
print("ahat", np.round(ahat, 4))
h = np.diag(XtX)/n; V = XtX/n
print("h (sample var)", np.round(h, 4))
s2A_diag = np.sum(h*alpha**2); s2A_full = alpha @ V @ alpha
print("true s2A diag", s2A_diag, "full", s2A_full)
print("plug diag", np.sum(h*ahat**2), "plug full", ahat @ V @ ahat)
Ainv = np.linalg.inv(XtX)
vif = np.diag(XtX)*np.diag(Ainv)
print("VIF", np.round(vif, 4), "sum", vif.sum())
print("bias full exact", L*sig**2/n, "bias diag exact", sig**2/n*vif.sum())
print("E plug full", s2A_full + L*sig**2/n, "E plug diag", s2A_diag + sig**2/n*vif.sum())
# residual variance and corrected estimator
res = y - X @ ahat; s2hat = res @ res/(n-L-1)
print("resid", np.round(res, 4), "s2hat (n-L-1 df)", s2hat, "corrected full", ahat @ V @ ahat - L*s2hat/n)
# The text fits no intercept (y = X alpha + eps), so the residual variance has n - L = 3 degrees of freedom
s2hat3 = res @ res/(n-L)
print("s2hat (n-L = 3 df, as in the text) %.5f  corrected full %.5f" % (s2hat3, ahat @ V @ ahat - L*s2hat3/n))
# Monte Carlo check
M = 200000
E = rng.standard_normal((M, n))*sig
Y = (X @ alpha)[None, :] + E
AH = np.linalg.solve(XtX, X.T @ Y.T).T
pf = np.einsum('ij,jk,ik->i', AH, V, AH); pdg = (AH**2*h).sum(1)
print("MC full", pf.mean(), "+-", pf.std()/np.sqrt(M), " MC diag", pdg.mean())
print("rank", np.linalg.matrix_rank(X), "cond", np.linalg.cond(XtX))
