import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import brentq
from scipy.stats import beta as Beta
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':9,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})
def Ek(y,tau):
    f=lambda u,p: 2*(1-u*u)**p*np.exp(-(1-u*u)*y*y/2)/(1-(1-tau*tau)*(1-u*u))
    return integrate.quad(lambda u:f(u,1),0,1,limit=200)[0]/integrate.quad(lambda u:f(u,0),0,1,limit=200)[0]
# tables
print('retention'); 
for tau in [0.5,0.18,0.05]: print(tau,[round(1-Ek(v,tau),3) for v in [0,2,3,4,6,8]])
S=json.load(open(str(_HERE / 'sims.json')))
print('threshold')
for r in S['table']: print(r['frac'],round(r['tau'],3),round(brentq(lambda y:1-Ek(y,r['tau'])-0.5,0.01,20),2))
# figure 7
fig,ax=plt.subplots(1,2,figsize=(5.4,2.2))
kk=np.linspace(0.002,0.998,500)
ax[0].plot(kk,np.ones_like(kk),label='Bayes Laplace, Beta(1,1)')
ax[0].plot(kk,Beta(0.5,0.5).pdf(kk),label='Jeffreys, horseshoe, Beta(1/2,1/2)')
ax[0].plot(kk,Beta(0.05,0.05).pdf(kk),label='near Haldane, Beta(0.05,0.05)')
ax[0].set_ylim(0,6); ax[0].set_xlabel(r'shrinkage weight $\kappa$'); ax[0].set_ylabel('prior density')
ax[0].set_title('the ladder of binomial priors',fontsize=9); ax[0].legend(fontsize=6.5,frameon=False,loc='upper center')
yy=np.linspace(0,8.5,90)
for tau in [0.5,0.18,0.05]:
    ax[1].plot(yy,[1-Ek(v,tau) for v in yy],label=r'$\tau=%g$'%tau)
ax[1].set_xlabel(r'$|y|$ (standard deviations)'); ax[1].set_ylabel('fraction retained')
ax[1].set_ylim(0,1.02); ax[1].set_title('the genome sets its own threshold',fontsize=9); ax[1].legend(fontsize=7,frameon=False)
fig.tight_layout(); fig.savefig(str(_FIGDIR) + '/ftns_f7_beta.pdf')
# figure 8
from ftns_rev_lib import fit, simulate
rng=np.random.default_rng(2024)
for frac,sd in [(0.10,3.0),(0.05,3.0),(0.02,2.5),(0.01,4.0)]:
    d,y=simulate(rng,5000,frac,sd)
G=fit(y); fig,ax=plt.subplots(1,3,figsize=(5.4,2.3))
ax[0].hist(G['Ek'],bins=40,color='C0'); ax[0].set_yscale('log'); ax[0].set_xlabel(r'$\mathrm{E}[\kappa_\ell\mid y]$'); ax[0].set_ylabel('loci')
ax[0].set_title('shrinkage weights, mean %.3f'%G['Ek'].mean(),fontsize=8)
e=np.sort(G['Ed']**2)[::-1]; c=np.cumsum(e)/e.sum(); x=100*np.arange(1,len(e)+1)/len(e)
ax[1].plot(x,100*c); ax[1].set_xlim(0,20); ax[1].set_ylim(0,102); ax[1].set_xlabel('percent of loci, ranked'); ax[1].set_ylabel('percent of estimated total')
ax[1].set_title('a few loci carry the total',fontsize=8)
sc=S['scaling']; Ls=[r['L'] for r in sc]; i=np.arange(len(Ls))
ax[2].bar(i-0.2,[r['truth'] for r in sc],0.4,label='truth')
ax[2].bar(i+0.2,[r['post'] for r in sc],0.4,yerr=[r['post_sdev'] for r in sc],capsize=2,label='horseshoe')
ax[2].scatter(i,[1.0]*len(i),marker='v',color='C3',zorder=3,label='Fisher: 0')
ax[2].set_yscale('log'); ax[2].set_ylim(0.8,1e8); ax[2].set_xticks(i); ax[2].set_xticklabels(['500','2k','5k','20k'],fontsize=7)
ax[2].set_xlabel(r'number of loci $L$'); ax[2].set_ylabel(r'$\sum_\ell\delta_\ell^2$'); ax[2].legend(fontsize=6,frameon=False,loc='upper left',ncol=1)
ax[2].set_title('the total is estimable',fontsize=8)
fig.tight_layout(w_pad=0.6); fig.savefig(str(_FIGDIR) + '/ftns_f8_genome.pdf')
