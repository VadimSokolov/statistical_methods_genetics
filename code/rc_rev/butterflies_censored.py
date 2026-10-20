"""Corbet's Malayan butterflies with the censoring of Fisher, Corbet and Williams (1943) taken
into account (round 2). The paper reports 620 species; 501 species (3306 individuals) fall in
the tabulated classes j = 1..24 and 119 are censored above 24, so n is not reported. Chapter
esf (code/esf_rev/butterflies_censored.py) profiles n out of the censored likelihood:
n hat = 11068, theta hat = 141.75. For fixed n the censored Ewens likelihood depends on theta
only through K = 620, as theta^620 Gamma(theta)/Gamma(theta+n), so the rc example conditions
on n = 11068 and K = 620 (a stated approximation, as in the sampler of Chapter esf).

Prints: MLE, Fisher information, se, posteriors under PH(0,1,2,1), Ga(0.1,0.1), Ga(2,1)
(log domain quadrature), predictive P(new), the normal approximation of the Ga(2,1) shift,
Good's a1/n, the half horseshoe Gibbs sampler of Example rc:ex:ewhh, the uncensored
comparison (n 3306, K 501), and the precision needed by the residue formula
(log10 kappa = log10 max|gamma_i| / C, computed in logarithms). The residue run itself is
code/rc_rev/butterflies_censored_residue.py."""
import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import brentq

rng = np.random.default_rng(20261020)
N_HAT, K_ALL = 11068, 620


def summarise(n, K, label):
    def loglik(t):
        return K * np.log(t) + gammaln(t) - gammaln(t + n)
    priors = {
        "PH(0,1,2,1)": lambda t: -np.log(t + 1) - np.log(t + 2),
        "Ga(0.1,0.1)": lambda t: -0.9 * np.log(t) - 0.1 * t,
        "Ga(2,1)": lambda t: np.log(t) - t,
    }
    grid = np.linspace(1e-6, 600, 600001)
    print("==", label, "n", n, "K", K)
    out = {}
    for name, lp in priors.items():
        lg = loglik(grid) + lp(grid)
        w = np.exp(lg - lg.max())
        cdf = np.cumsum(w); cdf /= cdf[-1]
        mean = np.sum(grid * w) / np.sum(w)
        q = [grid[np.searchsorted(cdf, p)] for p in (0.025, 0.5, 0.975)]
        pred = np.sum(grid / (grid + n) * w) / np.sum(w)
        out[name] = (mean, q[1], q[0], q[2], pred)
        print("%-12s mean %.2f median %.2f 95%% (%.2f, %.2f) P(new) %.5f" %
              (name, mean, q[1], q[0], q[2], pred))
    mle = brentq(lambda t: t * (digamma(t + n) - digamma(t)) - K, 1, 5000)
    i = np.arange(1, n + 1)
    info = np.sum((i - 1) / (mle * (mle + i - 1) ** 2))
    se = info ** -0.5
    print("MLE %.4f info %.5f se %.3f plug in P(new) %.5f a1/n %.4f" %
          (mle, info, se, mle / (mle + n), 118 / n))
    ga = out["Ga(2,1)"]
    print("Ga(2,1): median shift %.1f = %.2f se; normal approx shift (1/th - 1)/I = %.1f;"
          " theta hat - median %.1f; median/MLE %.3f" %
          (ga[1] - mle, (mle - ga[1]) / se, (1 / mle - 1) / info, mle - ga[1], ga[1] / mle))
    g01 = out["Ga(0.1,0.1)"]
    print("Ga(0.1,0.1) median shift %.2f = %.2f se" % (g01[1] - mle, (mle - g01[1]) / se))
    return mle, out


def ew_hh(n, k, chains=2000, iters=300, burn=100):
    """Escobar and West sampler with the half horseshoe, as in hh_gibbs.py."""
    th = np.full(chains, 1.0)
    keep = []
    for it in range(iters):
        tau = rng.exponential(1 / (1 + th)) + rng.exponential(1 / (2 + th))
        eta = rng.beta(th + 1, n)
        rate = tau - np.log(eta)
        odds = k / (n * rate)
        pick = rng.random(chains) < odds / (1 + odds)
        th = rng.gamma(np.where(pick, k + 1.0, float(k)), 1 / rate)
        if it >= burn:
            keep.append(th.copy())
    return np.concatenate(keep)


def log10_kappa(n, K, logC):
    """log10 of max_j |gamma_j| / C for the half horseshoe kernel theta^(K-1)/[(t+1)^2(t+2)^2
    prod_{i=3}^{n-1}(t+i)], whose simple poles at -j (3 <= j <= n-1) have residues
    j^(K-1) / ((j-1)^2 (j-2)^2 (j-3)! (n-1-j)!) in absolute value."""
    jj = np.arange(3, n)
    lg = ((K - 1) * np.log(jj) - 2 * np.log(jj - 1) - 2 * np.log(jj - 2)
          - gammaln(jj - 2) - gammaln(n - jj))
    return (lg.max() - logC) / np.log(10), jj[np.argmax(lg)]


if __name__ == "__main__":
    mle, out = summarise(N_HAT, K_ALL, "censored, conditional on n hat")
    summarise(3306, 501, "tabulated classes only (old, ignores the censoring)")
    s = ew_hh(N_HAT, K_ALL)
    print("sampler n hat: mean %.2f median %.2f 95%% (%.2f, %.2f) P(new) %.4f draws %d" %
          (s.mean(), np.median(s), *np.quantile(s, [0.025, 0.975]), np.mean(s / (s + N_HAT)), s.size))
    # normalizer C in logs for the half horseshoe kernel theta^K/[theta]^n /((t+1)(t+2))
    import mpmath as mp
    mp.mp.dps = 40
    g = lambda t: mp.exp(K_ALL * mp.log(t) + mp.loggamma(t) - mp.loggamma(t + N_HAT)) / ((t + 1) * (t + 2))
    pts = [0, 50, 100, 120, 130, 140, 150, 160, 180, 220, 400, mp.inf]
    C = mp.quad(g, pts, maxdegree=12)
    M = mp.quad(lambda t: t * g(t), pts, maxdegree=12)
    P = mp.quad(lambda t: t / (t + N_HAT) * g(t), pts, maxdegree=12)
    print("quad (mp, dps 40): C", mp.nstr(C, 15), "mean", mp.nstr(M / C, 15), "P(new)", mp.nstr(P / C, 15))
    # theta^K/[theta]^n = theta^(K-1) / prod_{i=1}^{n-1}(theta+i); with 1/((t+1)(t+2)) the
    # poles at -1, -2 are double, so N = n + 1 is the denominator degree
    lk, jmax = log10_kappa(N_HAT, K_ALL, float(mp.log(C)))
    N = N_HAT + 1
    print("log10 kappa %.1f at j=%d; N=%d; digits/N %.3f; ceil(N/2) %d" % (lk, jmax, N, lk / N, -(-N // 2)))
    lk0, _ = log10_kappa(3306, 501, float(mp.log(mp.mpf("1.89370975021e-9379"))))
    print("check old: log10 kappa %.1f (text: about 1794 digits), digits/N %.3f" % (lk0, lk0 / 3307))
    # finer quadrature for comparison with the residue run (butterflies_censored_residue_5500.txt)
    mp.mp.dps = 30
    pts = [0, 50, 80, 100] + list(range(105, 200, 3)) + [220, 260, 400, 1000, mp.inf]
    C2 = mp.quad(g, pts, maxdegree=14)
    M2 = mp.quad(lambda t: t * g(t), pts, maxdegree=14)
    P2 = mp.quad(lambda t: t / (t + N_HAT) * g(t), pts, maxdegree=14)
    print("fine quad (mp, dps 30): C", mp.nstr(C2, 15), "mean", mp.nstr(M2 / C2, 15), "P(new)", mp.nstr(P2 / C2, 15))
