import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng=np.random.default_rng(8888)
CB=["#1f77b4","#d62728","#2ca02c","#9467bd","#ff7f0e"]
def sticks(sig,th,K,reps):
    k=np.arange(1,K+1)
    B=rng.beta(1.0-sig, th+k*sig, size=(reps,K))
    return np.exp(np.cumsum(np.log(np.maximum(1-B,1e-300)),axis=1))
def freqs(R,G):
    Ra=G(R); p=np.empty_like(Ra); p[:,0]=1-Ra[:,0]; p[:,1:]=Ra[:,:-1]-Ra[:,1:]
    return np.clip(p,0,None)
def EK_and_A1(sig,th,G,ns,K=300000,reps=6,chunk=3):
    Kn=np.zeros(len(ns)); A1=np.zeros(len(ns)); done=0
    while done<reps:
        b=min(chunk,reps-done); done+=b
        p=freqs(sticks(sig,th,K,b),G)
        lq=np.log1p(-np.clip(p,0,1-1e-16))
        for i,n in enumerate(ns):
            Kn[i]+=(1-np.exp(n*lq)).sum(axis=1).sum()
            A1[i]+=(n*p*np.exp((n-1)*lq)).sum(axis=1).sum()
    return Kn/reps, A1/reps
sig,th=0.5,1.0
ns=np.array([1e3,1e4,1e5,1e6,1e7,1e8])

# ---- Figure A: the four regimes
weights=[(r"slowly varying, $\rho=0$", lambda r: 1/(1-0.5*np.log(np.maximum(r,1e-300))), CB[0]),
         (r"regularly varying, $\rho=2$", lambda r: np.maximum(r,1e-300)**2, CB[1]),
         (r"rapidly varying, $e^{-(1/r-1)}$", lambda r: np.exp(-(np.maximum(r,1e-300)**-1-1)), CB[2]),
         (r"vanishing near $0$", lambda r: np.clip((r-0.02)/0.98,0,1), CB[3])]
fig,ax=plt.subplots(1,2,figsize=(9.8,3.9))
for lab,G,c in weights:
    Kn,_=EK_and_A1(sig,th,G,ns)
    ax[0].plot(ns,Kn,"o-",color=c,lw=1.8,ms=4,label=lab)
    ax[1].plot(np.log(ns),Kn,"o-",color=c,lw=1.8,ms=4,label=lab)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("$n$"); ax[0].set_ylabel(r"$\mathbb{E}[K_n]$")
ax[0].legend(frameon=False,fontsize=8); ax[0].grid(alpha=.25,which="both")
ax[0].set_title("(a) log log scale: slope is the exponent",fontsize=10)
ax[1].set_xlabel(r"$\log n$"); ax[1].set_ylabel(r"$\mathbb{E}[K_n]$"); ax[1].set_yscale("log")
ax[1].grid(alpha=.25); ax[1].set_title(r"(b) against $\log n$: the slow regimes separate",fontsize=10)
fig.tight_layout(); fig.savefig(str(_FIGDIR) + "/py_fig-regimes.pdf"); plt.close(fig)

