"""Shared helpers for the pig chapter revision. Run scripts with python -I."""
import numpy as np
from scipy import special as sp, optimize

def loglik(theta, n, K):
    theta = np.asarray(theta, float)
    return K * np.log(theta) + sp.gammaln(theta) - sp.gammaln(theta + n)

def EK(theta, n):
    return theta * (sp.digamma(theta + n) - sp.digamma(theta))

def VarK(theta, n):
    i = np.arange(n)
    return np.sum(theta * i / (theta + i) ** 2)

def mle(n, K):
    if K >= n:
        return np.inf
    f = lambda p: EK(np.exp(p), n) - K
    return np.exp(optimize.brentq(f, -30, 30))

def info(theta, n):
    return VarK(theta, n) / theta ** 2

def grid_post(n, K, logprior_theta, lo=-12, hi=16, m=200001, extra=None):
    """Posterior on phi = log theta by quadrature. logprior_theta is log density in theta (Lebesgue)."""
    phi = np.linspace(lo, hi, m)
    th = np.exp(phi)
    lp = loglik(th, n, K) + logprior_theta(th) + phi  # Jacobian dtheta = theta dphi
    if extra is not None:
        lp = lp + extra(th)
    lp -= lp.max()
    w = np.exp(lp)
    w /= np.trapezoid(w, phi)
    return phi, th, w

def summ(phi, th, w):
    dphi = phi[1] - phi[0]
    p = w * dphi
    p = p / p.sum()
    mean = np.sum(th * p)
    sd = np.sqrt(np.sum((th - mean) ** 2 * p))
    skew = np.sum((th - mean) ** 3 * p) / sd ** 3
    c = np.cumsum(p)
    q = lambda a: np.interp(a, c, th)
    mode = th[np.argmax(w / th)]  # density in theta = w/theta
    return dict(mean=mean, sd=sd, skew=skew, median=q(0.5), lo=q(0.025), hi=q(0.975), mode=mode)

def gamma_lp(a, b):
    return lambda th: (a - 1) * np.log(th) - b * th

def pigprior_lp(n0, K0, shape_extra=0):
    # prior proportional to theta^(K0) Gamma(theta)/Gamma(theta+n0)
    return lambda th: (K0 + shape_extra) * np.log(th) + sp.gammaln(th) - sp.gammaln(th + n0)

def crp_K(theta, n, rng):
    i = np.arange(n)
    return int(np.sum(rng.random(n) < theta / (theta + i)))

def log_beta_draw(th, n, rng):
    """log of a Beta(th, n) draw, stable for tiny th: G1 = G(th+1) U^(1/th)."""
    lg1 = np.log(rng.gamma(th + 1.0)) + np.log(rng.random()) / th
    g2 = rng.gamma(n)
    return lg1 - np.logaddexp(lg1, np.log(g2))

def gibbs(n, K, a0, b0, T, rng, th0=1.0, burn=0):
    out = np.empty(T)
    th = th0
    for t in range(T):
        le = log_beta_draw(th, n, rng)
        th = rng.gamma(K + a0, 1.0 / (b0 - le))
        out[t] = th
    return out[burn:]

def iat(x, maxlag=200):
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0]
    s = 1.0
    for k in range(1, maxlag):
        if ac[k] < 0.05 and k > 5:
            break
        s += 2 * ac[k]
    return s, ac

STYLE = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
         "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
         "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42}
