"""All figures and simulation numbers for the revised pig chapter. python -I make_figures.py
Writes figures/pig_*.pdf and code/pig_rev/summary.json."""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np, sys, json, time
from piglib import *
from scipy import special as sp, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update(STYLE)
FIG = str(_FIGDIR) + "/"
OUT = {}
C1, C2, C3, C4 = "#1f3b5a", "#b5651d", "#2e7d5b", "#8b1e3f"
GA = gamma_lp(0.01, 0.01)
which = sys.argv[1:] or ["shape", "posterior", "mixing", "pig0", "goodturing", "predictive", "shakespeare", "fisherbayes", "butterflies"]

def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG + "pig_" + name + ".pdf"); plt.close(fig)

if "shape" in which:
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.3))
    for (n, K, lab, c) in [(200, 19, r"$n=200$, $K=19$", C1), (200, 4, r"$n=200$, $K=4$", C3), (30, 28, r"$n=30$, $K=28$", C4)]:
        phi, th, w = grid_post(n, K, GA, lo=-6, hi=10, m=40001)
        ax[0].plot(phi, w, color=c, lw=1.2, label=lab)
        m = mle(n, K); J = VarK(m, n)
        ax[0].plot(phi, stats.norm.pdf(phi, np.log(m), J ** -0.5), color=c, lw=0.9, ls=":")
    ax[0].set_xlabel(r"$\log\theta$"); ax[0].set_ylabel("posterior density")
    ax[0].set_title(r"(a) posterior of $\log\theta$", fontsize=8)
    ax[0].legend(frameon=False, fontsize=7)
    n = 20
    phi = np.linspace(-15, 14, 400001); th = np.exp(phi)
    cut = np.linspace(np.log(10), np.log(1e6), 300)
    for K, c in [(15, C1), (16, C3), (17, C2), (18, C4)]:
        lp = loglik(th, n, K) + phi; w = np.exp(lp - lp.max())
        Z = np.trapezoid(w, phi); cm = np.cumsum(th * w) * (phi[1] - phi[0]) / Z
        ax[1].plot(np.exp(cut), np.interp(cut, phi, cm), color=c, lw=1.2, label=fr"$K={K}$ ($n-K={n-K}$)")
        OUT[f"flat_K{K}"] = [float(np.interp(cc, phi, cm)) for cc in (6, 8, 10, 12)]
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"upper cutoff $c$ on $\theta$")
    ax[1].set_ylabel(r"$\int_0^c\theta\,p(\theta\mid K)\,d\theta$")
    ax[1].set_title(r"(b) flat prior, $n=20$", fontsize=8)
    ax[1].legend(frameon=False, fontsize=7)
    save(fig, "shape")

if "posterior" in which:
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.3))
    for k, (n, K, title) in enumerate([(1000, 27, r"(a) $n=1000$, $K=27$"), (15609, 240, r"(b) Williams, $n=15{,}609$, $K=240$")]):
        d = gibbs(n, K, 0.01, 0.01, 41000, np.random.default_rng(100 + k), th0=K / np.log(n), burn=1000)
        m = mle(n, K); se = info(m, n) ** -0.5
        OUT[f"post_{n}"] = dict(mean=d.mean(), sd=d.std(), mle=m, se=se)
        x = np.linspace(d.min(), np.quantile(d, 0.999), 300)
        ax[k].hist(d, bins=60, range=(x[0], x[-1]), density=True, color="#c9d3df", edgecolor="white", lw=0.2)
        ax[k].plot(x, stats.norm.pdf(x, m, se), color=C1, lw=1.2, label="Gaussian at MLE")
        ax[k].axvline(m, color=C4, ls="--", lw=0.9, label=f"MLE {m:.2f}")
        if n == 15609:
            ax[k].axvline(40.248, color=C2, ls=":", lw=1.0, label="Fisher 40.25")
        ax[k].set_xlabel(r"$\theta$"); ax[k].set_title(title, fontsize=8); ax[k].legend(frameon=False, fontsize=6, loc="upper left")
    ax[0].set_ylabel("density")
    save(fig, "posterior")

if "mixing" in which:
    n, K = 15609, 240
    T = 17000
    rng = np.random.default_rng(5)
    t0 = time.process_time(); dg = gibbs(n, K, 0.01, 0.01, T, rng, th0=40.0, burn=1000); tg = time.process_time() - t0
    # random walk Metropolis on log theta, proposal sd tuned to 1.0 * posterior sd of log theta
    lpost = lambda p: loglik(np.exp(p), n, K) + 0.01 * p - 0.01 * np.exp(p)
    rng = np.random.default_rng(6); p = np.log(40.0); lp = lpost(p); dr = np.empty(T); acc = 0
    t0 = time.process_time()
    for t in range(T):
        q = p + 0.07 * rng.standard_normal(); lq = lpost(q)
        if np.log(rng.random()) < lq - lp:
            p, lp = q, lq; acc += 1
        dr[t] = np.exp(p)
    tr = time.process_time() - t0; dr = dr[1000:]
    ig, acg = iat(dg); ir, acr = iat(dr)
    OUT["sampler"] = dict(g_mean=dg.mean(), g_sd=dg.std(), g_iat=ig, g_ess=len(dg) / ig, g_esss=len(dg) / ig / tg,
                          r_mean=dr.mean(), r_sd=dr.std(), r_iat=ir, r_ess=len(dr) / ir, r_esss=len(dr) / ir / tr, r_acc=acc / T)
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.3))
    L = 40
    ax[0].plot(range(L), acg[:L], "o-", ms=2.5, color=C1, lw=1, label=f"Beta augmentation Gibbs (IAT {ig:.1f})")
    ax[0].plot(range(L), acr[:L], "s-", ms=2.5, color=C2, lw=1, label=f"random walk Metropolis (IAT {ir:.1f})")
    ax[0].axhline(0, color="grey", lw=0.5); ax[0].set_xlabel("lag"); ax[0].set_ylabel("autocorrelation")
    ax[0].set_title(r"(a) autocorrelation of the $\theta$ chain, Williams", fontsize=8); ax[0].legend(frameon=False, fontsize=6.5)
    rng = np.random.default_rng(8); ms, pm = [], []
    for j in range(150):
        Kj = crp_K(6.0, 800, rng); ms.append(mle(800, Kj)); pm.append(summ(*grid_post(800, Kj, GA, m=20001))["mean"])
    ms, pm = np.array(ms), np.array(pm)
    OUT["mad150"] = float(np.mean(np.abs(ms - pm))); OUT["maxad150"] = float(np.max(np.abs(ms - pm)))
    ax[1].plot(ms, pm, "o", ms=2.5, color=C1, alpha=0.6); lim = [ms.min() * 0.95, ms.max() * 1.05]
    ax[1].plot(lim, lim, color=C4, ls="--", lw=0.8, label="$y=x$"); ax[1].legend(frameon=False)
    ax[1].set_xlabel(r"MLE $\hat\theta$"); ax[1].set_ylabel("posterior mean")
    ax[1].set_title(r"(b) 150 data sets, $\theta=6$, $n=800$", fontsize=8)
    save(fig, "mixing")
    # coverage, 400 data sets
    rng = np.random.default_rng(9); cov = covw = covl = 0; Ks = []
    for j in range(400):
        Kj = crp_K(6.0, 800, rng); Ks.append(Kj); s = summ(*grid_post(800, Kj, GA, m=20001))
        cov += s["lo"] <= 6 <= s["hi"]
        m = mle(800, Kj); se = info(m, 800) ** -0.5; J = VarK(m, 800)
        covw += m - 1.96 * se <= 6 <= m + 1.96 * se
        covl += abs(np.log(m) - np.log(6)) <= 1.96 / np.sqrt(J)
    OUT["coverage"] = dict(bayes=cov / 400, wald=covw / 400, logwald=covl / 400, meanK=float(np.mean(Ks)))

if "pig0" in which:
    rng = np.random.default_rng(10); Ktr = 1500; N = 200000
    kk = np.arange(1, Ktr + 1); X = np.zeros(N)
    for c in range(0, N, 2000):
        Y = 1.0 / rng.gamma(1.5, 1.0, size=(2000, Ktr)); X[c:c + 2000] = (Y / (4 * kk ** 2)).sum(1)
    tail = 0.5 * (np.pi ** 2 / 6 - np.sum(1.0 / kk ** 2)); X += tail
    ss = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
    est = np.array([np.mean(np.exp(-s ** 2 * X)) for s in ss]); ex = np.exp(-np.euler_gamma * ss) / sp.gamma(1 + ss)
    OUT["pig0"] = dict(s=ss.tolist(), est=est.tolist(), exact=ex.tolist(), rel=(np.abs(est / ex - 1)).tolist(), mean=X.mean(), tail=tail)
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.3))
    sg = np.linspace(0, 3.2, 200)
    ax[0].plot(sg, np.exp(-np.euler_gamma * sg) / sp.gamma(1 + sg), color=C1, lw=1.2, label=r"exact $e^{-\gamma s}/\Gamma(1+s)$")
    s2 = np.array([0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3])
    ax[0].plot(s2, [np.mean(np.exp(-s ** 2 * X)) for s in s2], "o", ms=3.5, color=C4, label="simulator")
    ax[0].set_xlabel("$s$"); ax[0].set_ylabel(r"$E[e^{-s^2X_0}]$"); ax[0].legend(frameon=False)
    ax[0].set_title("(a) reciprocal gamma transform", fontsize=8)
    # Fisher's nu from zero truncated negative binomial abundances, grid posterior
    S, eta0 = 400, 0.98
    def ztnb(nu, rng):
        out = []
        while len(out) < S:
            x = rng.negative_binomial(nu, 1 - eta0, size=4 * S); out.extend(x[x > 0].tolist())
        return np.array(out[:S])
    data = {"log series data (true $\\nu=0$)": np.random.default_rng(12).logseries(eta0, S),
            "negative binomial data (true $\\nu=0.5$)": ztnb(0.5, np.random.default_rng(13))}
    nu = np.linspace(1e-4, 1.2, 1200); eta = np.linspace(0.90, 0.9995, 800)
    NU, ET = np.meshgrid(nu, eta, indexing="ij")
    OUT["nu"] = {}
    for (lab, x), c in zip(data.items(), [C3, C1]):
        xs, cnt = np.unique(x, return_counts=True)
        ll = np.zeros_like(NU)
        lnorm = np.log(-np.expm1(NU * np.log1p(-ET)))
        for xi, ci in zip(xs, cnt):
            ll += ci * (sp.gammaln(xi + NU) - sp.gammaln(NU) - sp.gammaln(xi + 1) + NU * np.log1p(-ET) + xi * np.log(ET) - lnorm)
        w = np.exp(ll - ll.max()); pn = w.sum(1); pn /= np.trapezoid(pn, nu)
        cdf = np.cumsum(pn) * (nu[1] - nu[0])
        OUT["nu"][lab] = dict(mean=float(np.trapezoid(nu * pn, nu)), lo=float(np.interp(0.025, cdf, nu)), hi=float(np.interp(0.975, cdf, nu)),
                              K=int(len(x)), n=int(x.sum()))
        ax[1].plot(nu, pn, color=c, lw=1.2, label=("log series, $\\nu=0$" if "log" in lab else "neg. binomial, $\\nu=0.5$"))
    ax[1].axvline(0, color=C3, ls=":", lw=0.8); ax[1].axvline(0.5, color=C1, ls=":", lw=0.8)
    ax[1].set_xlabel(r"$\nu$"); ax[1].set_ylabel("posterior density"); ax[1].legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(0.56, 0.75))
    ax[1].set_title(r"(b) Fisher's shape $\nu$", fontsize=8); ax[1].set_xlim(-0.02, 1.0)
    save(fig, "pig0")

if "goodturing" in which:
    th, n, N = 3.0, 40, 60000
    rng = np.random.default_rng(14); acc = np.zeros(n + 1)
    for j in range(N):
        sizes = []
        for i in range(n):
            u = rng.random() * (i + th)
            if u >= i:
                sizes.append(1)
            else:
                c = np.searchsorted(np.cumsum(sizes), u, side="right"); sizes[c] += 1
        acc += np.bincount(sizes, minlength=n + 1)
    mc = acc / N
    mu = lambda r, n: th / r * np.exp(sp.gammaln(n + 1) - sp.gammaln(n - r + 1) + sp.gammaln(th + n - r) - sp.gammaln(th + n))
    OUT["good_mc"] = [[r, mc[r], mu(r, n), (r + 1) / (n + 1) * mu(r + 1, n + 1) / mu(r, n), r / (n + th)] for r in range(1, 7)]
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.3))
    R = np.arange(1, 9)
    ax[0].bar(R - 0.2, mc[R], width=0.4, color=C1, label="Monte Carlo")
    ax[0].bar(R + 0.2, [mu(r, n) for r in R], width=0.4, color=C2, label="closed form")
    ax[0].set_xlabel("$r$"); ax[0].set_ylabel(r"$E[a_r]$"); ax[0].legend(frameon=False)
    ax[0].set_title(r"(a) frequency of frequencies, $\theta=3$, $n=40$", fontsize=8)
    d = gibbs(15609, 240, 0.01, 0.01, 41000, np.random.default_rng(15), th0=40.0, burn=1000)
    q0 = d / (15609 + d); OUT["q0"] = dict(mean=q0.mean(), lo=np.quantile(q0, .025), hi=np.quantile(q0, .975))
    qr = {r: (r / (15609 + d)) for r in (1, 2, 5)}
    OUT["qr"] = {r: dict(mean=v.mean(), lo=np.quantile(v, .025), hi=np.quantile(v, .975), mlr=r / 15609) for r, v in qr.items()}
    ax[1].hist(q0 * 1e3, bins=60, density=True, color="#c9d3df", edgecolor="white", lw=0.2)
    for v in (OUT["q0"]["lo"], OUT["q0"]["hi"]):
        ax[1].axvline(v * 1e3, color=C4, ls=":", lw=0.9)
    ax[1].axvline(OUT["q0"]["mean"] * 1e3, color=C4, lw=1.0)
    ax[1].set_xlabel(r"unseen mass $q_0=\theta/(n+\theta)$, per thousand"); ax[1].set_ylabel("posterior density")
    ax[1].set_title("(b) Williams: posterior of the unseen mass", fontsize=8)
    save(fig, "goodturing")

if "predictive" in which:
    n0, K0, M = 3540, 178, 12069
    phi, th, w = grid_post(n0, K0, GA, lo=np.log(15), hi=np.log(90), m=301)
    wt = w / w.sum()
    Umax = 160; P = np.zeros((len(th), Umax + 1)); P[:, 0] = 1.0
    checkpoints = np.arange(0, 13001, 250); bands = []
    for i in range(n0, n0 + 13000 + 1):
        m = i - n0
        if m in set(checkpoints.tolist()):
            mix = wt @ P; c = np.cumsum(mix)
            bands.append([m, float(np.searchsorted(c, 0.025)), float(np.searchsorted(c, 0.5)), float(np.searchsorted(c, 0.975)), float(mix @ np.arange(Umax + 1))])
        if m == M:
            pmix = wt @ P
        p = (th / (th + i))[:, None]
        P[:, 1:] = P[:, 1:] * (1 - p) + P[:, :-1] * p; P[:, 0] *= (1 - p[:, 0])
    c = np.cumsum(pmix); u = np.arange(Umax + 1)
    OUT["pred"] = dict(mean=float(pmix @ u), sd=float(np.sqrt(pmix @ u ** 2 - (pmix @ u) ** 2)), lo=int(np.searchsorted(c, 0.025)), hi=int(np.searchsorted(c, 0.975)),
                       p_ge62=float(pmix[62:].sum()), plug=float(EK(39.355, n0 + M) - EK(39.355, n0)),
                       post_mean_theta=float(wt @ th), Ecomp=float(wt @ (th * (sp.digamma(th + n0 + M) - sp.digamma(th + n0)))),
                       Vsamp=float(wt @ np.array([np.sum(t * np.arange(n0, n0 + M) / (t + np.arange(n0, n0 + M)) ** 2) for t in th])),
                       Vpar=float(wt @ (th * (sp.digamma(th + n0 + M) - sp.digamma(th + n0))) ** 2 - (wt @ (th * (sp.digamma(th + n0 + M) - sp.digamma(th + n0)))) ** 2))
    fig, ax = plt.subplots(figsize=(3.8, 2.4))
    ax.bar(u, pmix, width=1.0, color="#c9d3df", edgecolor="white", lw=0.2)
    ax.axvline(62, color=C4, ls="--", lw=1, label="observed 62")
    ax.axvline(OUT["pred"]["plug"], color=C2, ls=":", lw=1, label="plug in 58.06")
    ax.axvline(OUT["pred"]["mean"], color=C1, lw=1, label=f"predictive mean {OUT['pred']['mean']:.1f}")
    ax.set_xlim(35, 85); ax.set_xlabel("number of new species in 12,069 further specimens"); ax.set_ylabel("predictive probability")
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "predictive")
    b = np.array(bands)
    fig, ax = plt.subplots(figsize=(3.8, 2.4))
    ax.fill_between(b[:, 0], b[:, 1], b[:, 3], color="#c9d3df", label="95% predictive band")
    ax.plot(b[:, 0], b[:, 4], color=C1, lw=1.1, label="predictive mean")
    ax.plot([M], [62], "o", color=C4, ms=4, label="observed, 1934 to 1936")
    ax.set_xlabel("further specimens $m$"); ax.set_ylabel("new species"); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    save(fig, "accumulation")
    OUT["acc_band_at_M"] = [list(x) for x in b if x[0] in (2000, 6000, 12000)]

if "shakespeare" in which:
    PP = pigprior_lp(10, 3, -1)
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.3))
    for (n, K, lab, c) in [(500, 240, r"unigrams, $n=500$, $K=240$", C1), (400, 396, r"trigrams, $n=400$, $K=396$", C4)]:
        phi, th, w = grid_post(n, K, PP, lo=2, hi=12, m=20001); ax[0].plot(phi, w, color=c, lw=1.2, label=lab)
    ax[0].set_xlabel(r"$\log\theta$"); ax[0].set_ylabel("posterior density"); ax[0].legend(frameon=False, fontsize=6.5)
    ax[0].set_title("(a) posterior under the pseudosample prior", fontsize=8)
    phi = np.linspace(2, 16, 4000); th = np.exp(phi)
    for (n, K, lab, c) in [(500, 240, r"unigrams, $n=500$" "\n" r"$K=240$", C1), (100, 100, r"trigrams" "\n" r"$n=K=100$", C4)]:
        ll = loglik(th, n, K); ax[1].plot(phi, ll - ll.max(), color=c, lw=1.2, label=lab)
    ax[1].set_ylim(-8, 0.5); ax[1].set_xlabel(r"$\log\theta$"); ax[1].set_ylabel("log likelihood minus maximum")
    ax[1].legend(frameon=False, fontsize=6.5, loc="lower right"); ax[1].set_title("(b) likelihood, no prior", fontsize=8)
    save(fig, "shakespeare")

if "fisherbayes" in which:
    n, k = 200, 25
    d = gibbs(n, k, 0.01, 0.01, 201000, np.random.default_rng(20260817), th0=7.0, burn=1000)
    m = mle(n, k); se = info(m, n) ** -0.5; med = np.median(d); lo, hi = np.quantile(d, [.025, .975])
    OUT["fisherbayes_gibbs"] = dict(mean=d.mean(), sd=d.std(), median=med, lo=lo, hi=hi, skew=float(stats.skew(d)))
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    ax.hist(d, bins=np.linspace(0, 20, 81), density=True, color="#c9d3df", edgecolor="white", lw=0.2, label="posterior, Gibbs draws")
    x = np.linspace(0, 20, 400); ax.plot(x, stats.norm.pdf(x, m, se), color=C4, lw=1.2, label=f"Gaussian at MLE {m:.2f}, SE {se:.2f}")
    ax.axvline(med, color=C1, ls="--", lw=0.9, label=f"posterior median {med:.2f}")
    for v in (lo, hi):
        ax.axvline(v, color=C1, ls=":", lw=0.8)
    for v in (m - 1.96 * se, m + 1.96 * se):
        ax.axvline(v, color=C4, ls=":", lw=0.8)
    ax.set_xlim(0, 20); ax.set_xlabel(r"$\theta$"); ax.set_ylabel("density"); ax.legend(frameon=False)
    save(fig, "fisherbayes")

if "butterflies" in which:
    a = np.array([118, 74, 44, 24, 29, 22, 20, 19, 20, 15, 12, 14, 6, 12, 6], float); r = np.arange(1, 16)
    TH = np.exp(np.linspace(np.log(60), np.log(250), 500)); X = np.linspace(0.85, 1.0, 1501)
    T, XX = np.meshgrid(TH, X, indexing="ij"); lp = np.zeros_like(T)
    for ri, ai in zip(r, a):
        mu = T * XX ** ri / ri; lp += ai * np.log(mu) - mu
    w = np.exp(lp - lp.max()); w /= w.sum()
    def q(f):
        v = f.ravel(); p = w.ravel(); o = np.argsort(v); c = np.cumsum(p[o]); return float(np.sum(v * p)), np.interp([0.025, 0.975], c, v[o]).tolist()
    fit = [q(T * XX ** ri / ri) for ri in r]
    rng = np.random.default_rng(21); idx = rng.choice(w.size, 20000, p=w.ravel())
    lam = (T * np.log1p(XX)).ravel()[idx]; U = rng.poisson(lam)
    OUT["butter"] = dict(theta=q(T), x=q(XX), newmean=q(T * np.log1p(XX)), U=[float(U.mean()), np.quantile(U, [.025, .975]).tolist()],
                         fit=[f[0] for f in fit], gt=((r[:-1] + 1) * a[1:] / a[:-1]).tolist(), smooth=[q(rr * XX) for rr in r[:-1]],
                         chi2=float(np.sum((a - np.array([f[0] for f in fit])) ** 2 / np.array([f[0] for f in fit]))))
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.3))
    ax[0].bar(r, a, color="#c9d3df", label="observed $a_r$")
    ax[0].errorbar(r, [f[0] for f in fit], yerr=np.array([[f[0] - f[1][0], f[1][1] - f[0]] for f in fit]).T, fmt="o", ms=2.5, color=C1, lw=0.8, label="log series fit")
    ax[0].set_xlabel("$r$"); ax[0].set_ylabel("number of species"); ax[0].legend(frameon=False)
    ax[0].set_title("(a) Corbet's butterflies, $r\\leq15$", fontsize=8)
    gt = (r[:-1] + 1) * a[1:] / a[:-1]
    ax[1].plot(r[:-1], gt, "s", ms=3, color=C2, label=r"raw Good Turing $r^*$")
    ax[1].plot(r[:-1], [s[0] for s in OUT["butter"]["smooth"]], "-", color=C1, lw=1.1, label=r"Bayesian smoothed $r\,x$")
    ax[1].plot(r[:-1], r[:-1], ":", color="grey", lw=0.8, label="$r^*=r$")
    ax[1].set_xlabel("$r$"); ax[1].set_ylabel("adjusted count $r^*$"); ax[1].legend(frameon=False, fontsize=6.5)
    ax[1].set_title("(b) adjusted counts", fontsize=8)
    save(fig, "butterflies")

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)): return o.item()
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o
import os
path = str(_HERE / "summary.json")
old = json.load(open(path)) if os.path.exists(path) else {}
old.update(clean(OUT)); json.dump(old, open(path, "w"), indent=1)
print(json.dumps(clean(OUT), indent=1))
