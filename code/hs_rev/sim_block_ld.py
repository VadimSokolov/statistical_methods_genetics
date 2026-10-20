"""Block linkage disequilibrium simulation of Section hs:sec:open (subsection
"Linkage disequilibrium changes the functional, not just the estimator") and
the linkage disequilibrium row of Table hs:tab:open in chapter hs.

Design: code/ftns/ftns_lib.design with rho_ld = 0.8 and bsize = 10, that is
L = 400 biallelic loci in blocks of ten whose dosages are thresholded from
latent normals with within block correlation 0.8, frequencies U(0.1, 0.9),
n = 300, twenty causal loci with a ~ N(0, 0.05^2), noise sd 0.10.  The truth is
sigma^2_A = a' V a with V the sample genotypic covariance.  Sixty replicates,
seeds 1 to 60 (the first twelve are also summarized
separately); the horseshoe chain (ftns_lib.horseshoe, 1200 iterations, 400
burn in) of replicate s uses default_rng(s + 5000).

For each replicate the script computes, as ratios to the truth,
  diagonal form of the truth             sum_l h_l a_l^2
  marginal plug in, diagonal weights   sum_l h_l am_l^2,     h = diag(V)
  marginal plug in, full form           am' V am
  horseshoe plug in, diagonal weights   sum_l h_l m_l^2,     m = posterior mean
  horseshoe posterior mean, diagonal    E[sum_l h_l alpha_l^2 | y]
  horseshoe plug in, full form          m' V m
  horseshoe posterior mean, full form   E[alpha' V alpha | y]
and prints their means over replicates with Monte Carlo standard errors.

This script was written in review round 2: the original run behind the
numbers first printed in the chapter left no script, so the chapter now quotes
this run.

Run:  python3 -I code/hs_rev/sim_block_ld.py      (under a minute on 4 cores)
Output cached in code/hs_rev/sim_block_ld.txt and sim_block_ld.json.
"""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FTNS = _ROOT / "code" / "ftns"
for _p in (str(_HERE), str(_FTNS)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import numpy as np
from multiprocessing import Pool
import ftns_lib as F

KEYS = ["truth_diag", "marg_diag", "marg_full", "hs_plug_diag", "hs_post_diag", "hs_plug_full", "hs_post_full"]


def run(seed):
    D = F.design(seed, L=400, n=300, ncausal=20, sig_e=0.10, rho_ld=0.8, bsize=10)
    X, y, V, s2A = D["X"], D["y"], D["V"], D["s2A"]
    h = np.diag(V).copy()
    am = (X.T @ y) / np.sum(X * X, 0)
    B = F.horseshoe(X, y, n_iter=1200, burn=400, rng=np.random.default_rng(seed + 5000))
    m = B.mean(0)
    out = dict(seed=seed, s2A=s2A, truth_diag=float(h @ D["a"] ** 2),
               marg_diag=float(h @ am ** 2), marg_full=float(am @ V @ am),
               hs_plug_diag=float(h @ m ** 2), hs_post_diag=float(((B ** 2) @ h).mean()),
               hs_plug_full=float(m @ V @ m), hs_post_full=float(np.einsum("ij,jk,ik->i", B, V, B).mean()))
    # mean within block correlation of the dosages, as a description of the design
    C = np.corrcoef(X.T)
    blk = np.arange(400) // 10
    same = (blk[:, None] == blk[None, :]) & ~np.eye(400, dtype=bool)
    out["r_within"] = float(np.nanmean(C[same]))
    return out


if __name__ == "__main__":
    with Pool(4) as pool:
        R = pool.map(run, range(1, 61))
    allsumm = {}
    for tag, RR in (("first12", R[:12]), ("all60", R)):
        tr = np.array([r["s2A"] for r in RR])
        summ = {"truth_mean": float(tr.mean()), "r_within": float(np.mean([r["r_within"] for r in RR]))}
        print(f"== {tag}: truth a'Va mean {tr.mean():.4f}; mean within block dosage correlation {summ['r_within']:.3f}")
        for k in KEYS:
            v = np.array([r[k] for r in RR]) / tr
            summ[k] = float(v.mean()); summ[k + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v)))
            print(f"{k:14s} ratio {v.mean():.3f}  (MC se {summ[k + '_se']:.3f})")
        for a, b in (("hs_plug_full", "hs_plug_diag"), ("hs_post_full", "hs_post_diag")):
            dlt = np.array([(r[a] - r[b]) / r["s2A"] for r in RR])
            print(f"paired difference {a} minus {b}: {dlt.mean():.3f} (MC se {dlt.std(ddof=1)/np.sqrt(len(dlt)):.3f})")
        allsumm[tag] = summ
    json.dump(dict(summary=allsumm, replicates=R), open(_HERE / "sim_block_ld.json", "w"), indent=1)
