# Stylized overdominance (sickle cell type) fitnesses: AA 0.9, AS 1.0, SS 0.2
import numpy as np
W = np.array([[0.9, 1.0], [1.0, 0.2]])   # allele order (A, S)
def summary(q):
    p = np.array([1-q, q]); wbar = p @ W @ p; wi = W @ p; al = wi - wbar
    s2A = 2*np.sum(p*al**2)
    fit = wbar + al[:, None] + al[None, :]
    s2D = np.sum(np.outer(p, p)*(W-fit)**2)
    dp = p*(wi-wbar)/wbar; pn = p+dp
    return dict(q=q, wbar=wbar, alpha=al, s2A=s2A, s2D=s2D, partial=s2A/wbar, exact=pn@W@pn-wbar, qnew=pn[1])
qstar = 0.1/0.9
for q in [0.01, 0.05, qstar]:
    print({k: (np.round(v, 6) if not isinstance(v, float) else round(v, 6)) for k, v in summary(q).items()})
print("qstar", qstar)
