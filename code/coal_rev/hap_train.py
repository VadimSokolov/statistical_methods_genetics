"""Quantile networks (2 x 128 ReLU, Adam, check loss on log theta) for four encoders, plus exact
posteriors given S and given H. Reports mean 90% width, coverage at 90 and 50, SBC chi square (10 bins).

Usage: python3 -I hap_train.py SEED EPOCHS KINDS PFX SHAPE SCALE, reading PFX_train.npz and PFX_test.npz
from the directory COAL_DATA_DIR (default: this directory), as written by hap_sim.py. The Ga(4, 25) block
of Table coal:tab:hap (round 2) is reproduced by slurm-free local runs recorded in hap_ga425.txt:
  python3 -I hap_sim.py 80000 2041 DIR/sat_train.npz 4 25
  python3 -I hap_sim.py 12000 2042 DIR/sat_test.npz 4 25
  COAL_DATA_DIR=DIR python3 -I hap_train.py 0 60 S,SH,xi,xiH sat 4 25"""
import sys, numpy as np, time
from scipy import stats
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from coalsim import a

D = _os.environ.get("COAL_DATA_DIR", str(_HERE)) + "/"   # where PFX_train.npz and PFX_test.npz live
PFX = sys.argv[4] if len(sys.argv) > 4 else "hap"
tr = np.load(D+PFX+"_train.npz"); te = np.load(D+PFX+"_test.npz")
n = 30
shape, scale = (float(sys.argv[5]), float(sys.argv[6])) if len(sys.argv) > 6 else (2.0, 4.0)

def feats(d, kind):
    xi = d["xi"]; S = xi.sum(1, keepdims=True)
    if kind == "S": return np.log1p(S)
    if kind == "xi": return np.log1p(xi)
    if kind == "xiH": return np.hstack([np.log1p(xi), np.log(d["H"])[:, None]])
    if kind == "xiHf": return np.hstack([np.log1p(xi), np.log(d["Hf"])[:, None]])
    if kind == "SH": return np.hstack([np.log1p(S), np.log(d["H"])[:, None]])

def train(Ztr, y, seed, epochs=40, bs=256, width=128):
    rng = np.random.default_rng(seed)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Z = (Ztr-mu)/sd
    d = Z.shape[1]+1
    W = [rng.normal(0, np.sqrt(2/d), (d, width)), rng.normal(0, np.sqrt(2/width), (width, width)), rng.normal(0, np.sqrt(1/width), (width, 1))]
    b = [np.zeros(width), np.zeros(width), np.array([y.mean()])]
    params = W + b
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    t = 0; N = len(y); steps = epochs*(N//bs)
    for ep in range(epochs):
        perm = rng.permutation(N)
        for k in range(N//bs):
            idx = perm[k*bs:(k+1)*bs]
            u = rng.uniform(0.005, 0.995, bs)
            x0 = np.hstack([Z[idx], np.log(u[:, None]/(1-u[:, None]))/2])
            h1 = x0@W[0]+b[0]; a1_ = np.maximum(h1, 0)
            h2 = a1_@W[1]+b[1]; a2_ = np.maximum(h2, 0)
            out = (a2_@W[2]).ravel()+b[2]
            r = y[idx]-out
            g = -(u-(r < 0))/bs           # d loss / d out
            gW2 = a2_.T@g[:, None]; gb2 = np.array([g.sum()])
            d2 = (g[:, None]@W[2].T)*(h2 > 0)
            gW1 = a1_.T@d2; gb1 = d2.sum(0)
            d1 = (d2@W[1].T)*(h1 > 0)
            gW0 = x0.T@d1; gb0 = d1.sum(0)
            grads = [gW0, gW1, gW2, gb0, gb1, gb2]
            t += 1; lr = 1e-3*(0.05**(t/steps))
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9*m[i]+0.1*gr; v[i] = 0.999*v[i]+0.001*gr**2
                p -= lr*(m[i]/(1-0.9**t))/(np.sqrt(v[i]/(1-0.999**t))+1e-8)
    def predict(Zte, u):
        Zs = (Zte-mu)/sd
        outs = []
        for uu in u:
            x0 = np.hstack([Zs, np.full((len(Zs), 1), np.log(uu/(1-uu))/2)])
            a1_ = np.maximum(x0@W[0]+b[0], 0); a2_ = np.maximum(a1_@W[1]+b[1], 0)
            outs.append((a2_@W[2]).ravel()+b[2])
        return np.sort(np.array(outs).T, 1)
    return predict

levels = np.arange(1, 100)/100.0
def summarise(name, Q, th):
    # Q: quantiles of theta at levels (sorted), shape (M, 99)
    i05, i25, i75, i95 = 4, 24, 74, 94
    w = np.mean(Q[:, i95]-Q[:, i05])
    c90 = np.mean((th > Q[:, i05]) & (th < Q[:, i95])); c50 = np.mean((th > Q[:, i25]) & (th < Q[:, i75]))
    ranks = (Q < th[:, None]).sum(1)          # 0..99
    cnt = np.bincount(np.minimum(ranks//10, 9), minlength=10)  # 0..9 / 10..19 / ... / 90..99
    exp = len(th)*np.array([0.10]*9+[0.10])
    # exact expected bin probabilities under uniform PIT with 99 levels: ranks 0..9 -> P(PIT<0.10) etc.
    chi = ((cnt-exp)**2/exp).sum()
    print("%-22s width %.3f cov90 %.3f cov50 %.3f chi2 %.1f" % (name, w, c90, c50, chi))
    return w, c90, c50, chi

th_te = te["th"]
res = {}
# exact posterior given S
grid = np.linspace(1e-3, 70 if shape*scale < 20 else 500, 7000); smax = int(te["xi"].sum(1).max())+5
P = np.zeros((smax+1, grid.size)); P[0] = 1
for k in range(2, n+1):
    p = (k-1)/(k-1+grid); q = 1-p
    new = np.empty_like(P); new[0] = p*P[0]
    for s in range(1, smax+1): new[s] = q*new[s-1]+p*P[s]
    P = new
prior = stats.gamma.pdf(grid, shape, scale=scale)
post = P*prior; cdf = np.cumsum(post, 1); cdf /= cdf[:, -1:]
S_te = te["xi"].sum(1)
Q = np.array([np.interp(levels, cdf[s], grid) for s in S_te])
res["exact S"] = summarise("exact given S", Q, th_te)
# exact posterior given H (Ewens): lik theta^H / rising factorial theta^(n)
logrf = sum(np.log(grid+i) for i in range(n))
cdfH = {}
for h in np.unique(te["H"]):
    lp = h*np.log(grid)-logrf+np.log(prior+1e-300); w_ = np.exp(lp-lp.max()); c = np.cumsum(w_); cdfH[h] = c/c[-1]
Q = np.array([np.interp(levels, cdfH[h], grid) for h in te["H"]])
res["exact H"] = summarise("exact given H", Q, th_te)
Qp = np.tile(stats.gamma.ppf(levels, shape, scale=scale), (len(th_te), 1))
summarise("prior", Qp, th_te)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 40
KINDS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["S", "xi", "xiH", "xiHf", "SH"]
for kind in KINDS:
    t0 = time.time()
    pred = train(feats(tr, kind), np.log(tr["th"]), seed=seed, epochs=EP)
    Q = np.exp(pred(feats(te, kind), levels))
    res[kind] = summarise("net "+kind, Q, th_te)
    print("   train time %.1f" % (time.time()-t0))
if _os.environ.get("COAL_RES_JSON"):
    import json
    json.dump({k: [float(x) for x in v] for k, v in res.items()}, open(_os.environ["COAL_RES_JSON"], "w"), indent=1)
