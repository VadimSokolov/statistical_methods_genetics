"""Section 3: Polya urn, Dirichlet, stick breaking; figure toolkit_stick.pdf."""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np
from fractions import Fraction
from scipy import integrate, special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Polya urn: 1 black 1 white, add one of the colour drawn. P(B,B,W)
def seqprob(seq, b=1, w=1):
    pr = Fraction(1)
    for x in seq:
        if x == 'B': pr *= Fraction(b, b+w); b += 1
        else: pr *= Fraction(w, b+w); w += 1
    return pr
print("P(BBW)", seqprob("BBW"), "P(BWB)", seqprob("BWB"), "P(WBB)", seqprob("WBB"),
      "int p^2(1-p)", integrate.quad(lambda p: p*p*(1-p), 0, 1)[0])
# general urn b=2,w=1 : P(BBW) vs Beta(2,1) moment
print("b=2,w=1 P(BBW)", seqprob("BBW",2,1), "Beta(2,1) moment", special.beta(4,2)/special.beta(2,1))
# number of black after n draws uniform
n=5
from itertools import product
cnt = {}
for seq in product("BW", repeat=n):
    k = seq.count('B'); cnt[k] = cnt.get(k,0)+seqprob(seq)
print("distribution of blacks in 5 draws", cnt)
rng = np.random.default_rng(1)
# simulate limiting fraction for b=2,w=1 after 2000 draws
fr = []
for r in range(4000):
    b, w = 2, 1
    u = rng.random(2000)
    for x in u:
        if x < b/(b+w): b += 1
        else: w += 1
    fr.append(b/(b+w))
fr = np.array(fr)
print("urn limit mean var", fr.mean().round(4), fr.var().round(4), "Beta(2,1)", 2/3, (2*1)/(9*4))

# Dirichlet aggregation check: (P1+P2, P3) ~ Beta(a1+a2,a3)
a = np.array([1.,2.,3.])
P = rng.dirichlet(a, 200000)
print("Dirichlet agg mean", (P[:,0]+P[:,1]).mean().round(4), "pred", 3/6, "var", (P[:,0]+P[:,1]).var().round(5), "pred", 3*3/(36*7))

# GEM expected weights and tails
def gem_mean(alpha, theta, K):
    j = np.arange(1, K+1)
    EW = (1-alpha)/(1+theta+(j-1)*alpha)
    tail = np.cumprod(1-EW)
    EP = EW*np.concatenate([[1], tail[:-1]])
    return EP, tail
for (al, th) in [(0,5),(0.5,5)]:
    EP, tail = gem_mean(al, th, 2000)
    k99 = np.argmax(tail < 0.01)+1
    print("GEM", al, th, "EP1", EP[0].round(4), "EP2", EP[1].round(4), "atoms to leave <1% mass (expected)", k99, "tail at 25", tail[24].round(4))

def gem_draw(alpha, theta, K, rng):
    j = np.arange(1, K+1)
    W = rng.beta(1-alpha, theta+j*alpha)
    return W*np.concatenate([[1], np.cumprod(1-W)[:-1]])

plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":9,
                     "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3), sharey=True)
K = 40; rng = np.random.default_rng(7)
for ax, (al, th) in zip(axes, [(0,5),(0.5,5)]):
    draws = gem_draw(al, th, K, rng)
    EP, _ = gem_mean(al, th, K)
    ax.bar(np.arange(1,K+1), draws, color="#7a9cc6", width=0.8, label="one realisation")
    ax.plot(np.arange(1,K+1), EP, color="#b5523b", lw=1.2, label="expected weight")
    ax.set_yscale("log"); ax.set_ylim(1e-5, 1)
    ax.set_xlabel("stick index $j$")
    ax.set_title(r"$\mathrm{GEM}(%g,\,%g)$" % (al, th))
axes[0].set_ylabel("weight $P_j$")
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(str(_FIGDIR) + "/toolkit_stick.pdf")
