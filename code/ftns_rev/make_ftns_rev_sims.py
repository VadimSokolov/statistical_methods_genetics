import numpy as np, json
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from ftns_rev_lib import fit, simulate
out={}
rng=np.random.default_rng(2024); rows=[]
for frac,sd in [(0.10,3.0),(0.05,3.0),(0.02,2.5),(0.01,4.0)]:
    d,y=simulate(rng,5000,frac,sd); P=fit(y); sq=np.sort(P['E2'])[::-1]
    rows.append(dict(frac=frac,sd=sd,truth=float((d*d).sum()),unshrunk=float((y*y).sum()-5000),hs=float(P['total']),
        plug=float(P['plug']),tau=float(P['tau']),meank=float(P['Ek'].mean()),top5=float(sq[:250].sum()/sq.sum())))
    print(rows[-1],flush=True)
    if frac==0.01: np.savez(str(_HERE / 'genome_example.npz'),Ek=P['Ek'],E2=P['E2'])
out['table']=rows
rng=np.random.default_rng(7); sc=[]
for L,R in [(500,40),(2000,40),(5000,30),(20000,20)]:
    o=[]
    for rep in range(R):
        d,y=simulate(rng,L,0.05,3.0); P=fit(y); t=(d*d).sum()
        o.append((t,P['total'],P['total_sd'],P['plug'],(y*y).sum()-L,abs(P['total']-t)<1.96*P['total_sd']))
    o=np.array(o,float); e=o[:,1]-o[:,0]
    r=dict(L=L,truth=o[:,0].mean(),post=o[:,1].mean(),mcsd=e.std(),relmcsd=(e/o[:,0]).std(),postsd=o[:,2].mean(),
           relbias=(e/o[:,0]).mean(),plugbias=((o[:,3]-o[:,0])/o[:,0]).mean(),unsh_relsd=((o[:,4]-o[:,0])/o[:,0]).std(),cover=o[:,5].mean(),
           post_sdev=o[:,1].std())
    sc.append(r); print(r,flush=True)
out['scaling']=sc
json.dump(out,open(str(_HERE / 'sims.json'),'w'),indent=1)
