# Redraw f1 (geometry) and f4 (errors) without hyphenated figure text
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':9,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})
# f4
n=np.logspace(1,5.6,200)
fig,ax=plt.subplots(figsize=(3.8,2.6))
ax.loglog(n,6.38/np.sqrt(n),color='#1f4e79',lw=1.8,label='sampling noise')
ax.loglog(n,10.19/n,color='#d4880f',lw=1.8,label='plug in bias')
ax.loglog(n,0.0905*np.ones_like(n),color='#c0392b',lw=1.5,label=r"Fisher's error $\mathcal{R}$")
for v,t in [(113,r'$n=113$'),(4975,r'$n^\ast=4975$')]:
    ax.axvline(v,ls=':',lw=0.8,color='grey'); ax.text(v*1.1,2e-3,t,fontsize=7,color='grey')
ax.set_xlabel(r'$n$ per genotype'); ax.set_ylabel('relative size'); ax.set_title(r"Fisher's error is the largest term only past $n^\ast$",fontsize=9)
ax.legend(fontsize=7,frameon=False,loc='lower left'); fig.tight_layout(); fig.savefig(str(_FIGDIR) + '/ftns_f4_errors.pdf')
# f1
fig,ax=plt.subplots(figsize=(5.4,2.7)); ax.axis('off')
x=np.linspace(0,10,300); y=0.5*np.sin(x*0.62+0.3)
ax.fill_between(x,y,-1.6,color='#e8f1f4'); ax.plot(x,y,color='#178a8a',lw=2)
x0=2.3; y0=0.5*np.sin(x0*0.62+0.3); sl=0.5*0.62*np.cos(x0*0.62+0.3)
t=np.array([1,sl]); t=t/np.linalg.norm(t)*2.2; nrm=np.array([-t[1],t[0]])/np.linalg.norm(t)*1.6
ax.annotate('',xy=(x0+t[0]+nrm[0],y0+t[1]+nrm[1]),xytext=(x0,y0),arrowprops=dict(arrowstyle='-|>',lw=2,color='k'))
ax.annotate('',xy=(x0+t[0],y0+t[1]),xytext=(x0,y0),arrowprops=dict(arrowstyle='-|>',lw=1.6,color='#1f4e79'))
ax.annotate('',xy=(x0+t[0]+nrm[0],y0+t[1]+nrm[1]),xytext=(x0+t[0],y0+t[1]),arrowprops=dict(arrowstyle='-|>',lw=1.6,color='#c0392b'))
ax.plot([x0],[y0],'ko',ms=5); ax.text(x0-0.8,y0-0.35,r'$P(t)$')
ax.text(x0+t[0]+nrm[0]+0.1,y0+t[1]+nrm[1]+0.05,r'$\dot P$')
ax.text(x0+t[0]+0.15,y0+t[1]+0.6,'transverse\n'+r'$\Rightarrow\ \mathcal{R}$',color='#c0392b',fontsize=8)
ax.text(x0+0.9,y0-0.8,'tangential\n'+r'$\Rightarrow\ \sigma^2_A$',color='#1f4e79',fontsize=8)
ax.text(0.1,-1.15,r'$\mathcal{M}$: gene frequency product measures',color='#178a8a',fontsize=8)
ax.text(0.1,-1.45,'(Hardy Weinberg proportions and linkage equilibrium)',color='#178a8a',fontsize=7)
ax.set_title(r"Fisher's assertion: the genotype distribution never leaves $\mathcal{M}$",fontsize=9)
ax.set_xlim(0,10); ax.set_ylim(-1.6,2.6); fig.tight_layout(); fig.savefig(str(_FIGDIR) + '/ftns_f1_geometry.pdf')
