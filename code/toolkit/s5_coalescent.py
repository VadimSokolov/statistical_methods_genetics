"""Section 5: Kingman coalescent, Watterson, site frequency spectrum; figure toolkit_coalescent.pdf."""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def an(n): return sum(1/k for k in range(1, n))
def bn(n): return sum(1/k**2 for k in range(1, n))
for n in [2, 5, 10, 11, 100]:
    print("n", n, "E Tmrca", round(2*(1-1/n),4), "E L", round(2*an(n),4), "Var L", round(4*bn(n),4),
          "Var Tmrca", round(sum(4/(k*(k-1))**2 for k in range(2,n+1)),4))

rng = np.random.default_rng(11)
def sim_sfs(n, theta, R, rng):
    xi = np.zeros((R, n-1)); S = np.zeros(R); H = np.zeros(R); L = np.zeros(R)
    for r in range(R):
        sizes = [1]*n           # leaf counts below each lineage
        t = 0.0
        while len(sizes) > 1:
            k = len(sizes)
            T = rng.exponential(2/(k*(k-1)))
            t += T
            # mutations on each branch during this epoch
            m = rng.poisson(theta/2*T, size=k)
            for s, mm in zip(sizes, m):
                xi[r, s-1] += mm
            L[r] += k*T
            i, j = rng.choice(k, 2, replace=False)
            new = sizes[i]+sizes[j]
            sizes = [sizes[x] for x in range(k) if x not in (i, j)] + [new]
        H[r] = t
    return xi, L, H

n, theta, R = 5, 2.0, 100000
xi, L, H = sim_sfs(n, theta, R, rng)
S = xi.sum(1)
print("n=5 theta=2 E xi sim", xi.mean(0).round(3), "theory", [round(theta/i,3) for i in range(1,n)])
print(" E S sim", S.mean().round(3), "theory", round(theta*an(n),4), "Var S sim", S.var().round(3),
      "theory", round(theta*an(n)+theta**2*bn(n),4))
print(" E L", L.mean().round(4), 2*an(n), "E H", H.mean().round(4), 2*(1-1/n))
print(" Var L", L.var().round(4), 4*bn(n))
# P(S=0) = prod (k-1)/(k-1+theta)
print(" P(S=0) sim", (S==0).mean().round(4), "theory", round(np.prod([(k-1)/(k-1+theta) for k in range(2,n+1)]),4))

# Kreitman Adh: n=11, S=43
n, S = 11, 43
a = an(n); b = bn(n)
thW = S/a
print("Kreitman a_n", round(a,4), "b_n", round(b,4), "theta_W", round(thW,3),
      "sd(S) at theta_W", round(np.sqrt(thW*a+thW**2*b),2), "sd(theta_W)", round(np.sqrt(thW*a+thW**2*b)/a,2))
print("expected SFS at theta_W n=11:", [round(thW/i,2) for i in range(1,n)])

# Figure: one realisation of the coalescent, n=8, theta=3, mutations as dots
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":9,
                     "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
def build(seed, n=8, theta=3.0):
    rng = np.random.default_rng(seed)
    nodes = {i: dict(t=0.0, kids=[], leaves=[i]) for i in range(n)}
    active = list(range(n)); nxt = n; t = 0.0; epochs = []
    while len(active) > 1:
        k = len(active); T = rng.exponential(2/(k*(k-1))); t += T
        epochs.append((k, t))
        i, j = rng.choice(len(active), 2, replace=False)
        a_, b_ = active[i], active[j]
        nodes[nxt] = dict(t=t, kids=[a_, b_], leaves=nodes[a_]['leaves']+nodes[b_]['leaves'])
        active = [x for x in active if x not in (a_, b_)] + [nxt]; nxt += 1
    muts = {}
    for v, d in nodes.items():
        for c in d['kids']:
            m = rng.poisson(theta/2*(d['t']-nodes[c]['t']))
            muts[c] = np.sort(rng.uniform(nodes[c]['t'], d['t'], m))
    return nodes, active[0], epochs, muts

n, theta = 8, 3.0
for seed in range(2000):
    nodes, root, epochs, muts = build(seed, n, theta)
    S_ = sum(len(m) for m in muts.values())
    top = sum(len(muts[c]) for c in nodes[root]['kids'])
    if 1.5 < nodes[root]['t'] < 2.1 and 7 <= S_ <= 10 and top > 0:
        break
print("figure seed", seed)
order = []
def dfs(v):
    if not nodes[v]['kids']: order.append(v); return
    for c in nodes[v]['kids']: dfs(c)
dfs(root)
xpos = {v: i for i, v in enumerate(order)}
def setx(v):
    if nodes[v]['kids']:
        xpos[v] = np.mean([setx(c) for c in nodes[v]['kids']])
    return xpos[v]
setx(root)
fig, ax = plt.subplots(figsize=(3.8, 2.8))
mutcount = np.zeros(n-1, int)
for v, d in nodes.items():
    for c in d['kids']:
        x0, t0, t1 = xpos[c], nodes[c]['t'], d['t']
        ax.plot([x0, x0], [t0, t1], color="#333333", lw=1)
        ys = muts[c]
        if len(ys):
            ax.plot([x0]*len(ys), ys, 'o', ms=3.2, color="#b5523b", zorder=3)
            mutcount[len(nodes[c]['leaves'])-1] += len(ys)
    if d['kids']:
        xs = [xpos[c] for c in d['kids']]
        ax.plot([min(xs), max(xs)], [d['t'], d['t']], color="#333333", lw=1)
for (k, tk) in epochs:
    ax.axhline(tk, color="#bbbbbb", lw=0.5, ls=":", zorder=0)
ax.set_xticks(range(n)); ax.set_xticklabels([str(i+1) for i in range(n)])
ax.set_xlabel("sampled genes"); ax.set_ylabel("time (units of $N$ generations)")
ax.spines['bottom'].set_visible(False); ax.tick_params(axis='x', length=0)
fig.tight_layout()
fig.savefig(str(_FIGDIR) + "/toolkit_coalescent.pdf")
L = sum(nodes[v]['t']-nodes[c]['t'] for v in nodes for c in nodes[v]['kids'])
print("figure: height", round(nodes[root]['t'],3), "length", round(L,3), "S", int(mutcount.sum()),
      "spectrum xi_1..xi_7", mutcount.tolist(), "epoch times", [round(t,3) for _,t in epochs])
# exact identity: E L_i = sum_k k E T_k P(lineage subtends i) = 2/i
from math import comb
for n in [5, 11, 30]:
    vals = [sum(2/(k-1)*comb(n-i-1,k-2)/comb(n-1,k-1) for k in range(2, n-i+2)) for i in range(1,n)]
    print("n", n, "max |E L_i - 2/i|", max(abs(v-2/i) for i, v in zip(range(1,n), vals)))
