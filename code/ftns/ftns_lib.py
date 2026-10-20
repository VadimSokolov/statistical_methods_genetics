import numpy as np, json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "font.family": "serif"})
t0 = time.time()


# ======================================================================
# samplers
# ======================================================================
def _bcm(X, y, D, sig2, rng):
    """Draw from N(A^{-1}Phi'ytil, A^{-1}), A = Phi'Phi + D^{-1}, Phi=X/sig."""
    n, p = X.shape
    Phi = X/np.sqrt(sig2); alph = y/np.sqrt(sig2)
    u = rng.standard_normal(p)*np.sqrt(D)
    v = Phi @ u + rng.standard_normal(n)
    M = (Phi*D) @ Phi.T + np.eye(n)
    w = np.linalg.solve(M, alph - v)
    return u + D*(Phi.T @ w)


def horseshoe(X, y, n_iter=1200, burn=400, rng=None, regularized=False):
    rng = rng or np.random.default_rng(0)
    n, p = X.shape
    beta = np.zeros(p); lam2 = np.ones(p); tau2 = 1.0
    nu = np.ones(p); xi = 1.0; sig2 = float(np.var(y)); c2 = 4.0
    keep = np.zeros((n_iter-burn, p))
    for it in range(n_iter):
        l2 = lam2*c2/(c2 + tau2*lam2) if regularized else lam2
        D = sig2*tau2*l2
        beta = _bcm(X, y, D, sig2, rng)
        r = y - X @ beta
        sig2 = 1.0/rng.gamma((n+p)/2.0,
                             1.0/(0.5*r @ r + 0.5*np.sum(beta**2/(tau2*l2))))
        lam2 = 1.0/rng.gamma(1.0, 1.0/(1.0/nu + beta**2/(2*tau2*sig2)))
        nu = 1.0/rng.gamma(1.0, 1.0/(1.0 + 1.0/lam2))
        tau2 = 1.0/rng.gamma((p+1)/2.0,
                             1.0/(1.0/xi + np.sum(beta**2/lam2)/(2*sig2)))
        xi = 1.0/rng.gamma(1.0, 1.0/(1.0 + 1.0/tau2))
        if regularized:
            c2 = 1.0/rng.gamma(2.0 + p/2.0, 1.0/(8.0 + np.sum(beta**2/(2*sig2*tau2*lam2))))
        if it >= burn:
            keep[it-burn] = beta
    return keep


def bivariate_horseshoe(X, yw, yz, n_iter=1200, burn=400, rng=None):
    """Two responses on the same design, correlated residuals, shared local scales."""
    rng = rng or np.random.default_rng(0)
    n, p = X.shape
    a = np.zeros(p); b = np.zeros(p)
    lam2 = np.ones(p); nu = np.ones(p); tau2 = 1.0; xi = 1.0
    sw = float(np.var(yw)); sz = float(np.var(yz)); rho = 0.0
    A = np.zeros((n_iter-burn, p)); B = np.zeros((n_iter-burn, p))
    for it in range(n_iter):
        # alpha | beta : condition the w-residual on the z-residual
        rz = yz - X @ b
        cw = sw*(1-rho**2)
        yw_adj = yw - rho*np.sqrt(sw/sz)*rz
        a = _bcm(X, yw_adj, cw*tau2*lam2, cw, rng)
        rw = yw - X @ a
        cz = sz*(1-rho**2)
        yz_adj = yz - rho*np.sqrt(sz/sw)*rw
        b = _bcm(X, yz_adj, cz*tau2*lam2, cz, rng)
        # residual covariance
        rw = yw - X @ a; rz = yz - X @ b
        sw = float(rw @ rw)/n; sz = float(rz @ rz)/n
        rho = float(np.clip((rw @ rz)/(n*np.sqrt(sw*sz)), -0.95, 0.95))
        # shared local scales: two coefficients per locus
        q = (a**2/cw + b**2/cz)/(2*tau2)
        lam2 = 1.0/rng.gamma(1.5, 1.0/(1.0/nu + q))
        nu = 1.0/rng.gamma(1.0, 1.0/(1.0 + 1.0/lam2))
        tau2 = 1.0/rng.gamma((2*p+1)/2.0,
                             1.0/(1.0/xi + np.sum((a**2/cw + b**2/cz)/lam2)/2.0))
        xi = 1.0/rng.gamma(1.0, 1.0/(1.0 + 1.0/tau2))
        if it >= burn:
            A[it-burn] = a; B[it-burn] = b
    return A, B


# ======================================================================
# design
# ======================================================================
def design(seed, L=400, n=300, ncausal=20, sig_e=0.10, rho_ld=0.0, bsize=10):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, L)
    if rho_ld > 0:                       # block LD via correlated latent normals
        Z = np.empty((n, L))
        for s in range(0, L, bsize):
            e = L - s if L - s < bsize else bsize
            f = rng.standard_normal((n, 1))
            Z[:, s:s+e] = np.sqrt(rho_ld)*f + np.sqrt(1-rho_ld)*rng.standard_normal((n, e))
        from scipy.stats import norm
        c1 = norm.ppf((1-p)**2); c2 = norm.ppf((1-p)**2 + 2*p*(1-p))
        G = (Z > c1).astype(float) + (Z > c2).astype(float)
    else:
        G = rng.binomial(2, p, size=(n, L)).astype(float)
    causal = rng.choice(L, ncausal, replace=False)
    a = np.zeros(L); a[causal] = rng.standard_normal(ncausal)*0.05
    Xc = G - G.mean(0)
    w = 1.0 + Xc @ a
    y = w - w.mean() + rng.standard_normal(n)*sig_e
    V = np.cov(G.T)                      # genotypic covariance: sigma^2_A = a' V a
    s2A = float(a @ V @ a)
    return dict(rng=rng, X=Xc, y=y, a=a, causal=causal, s2A=s2A, V=V, p=p, w=w)


