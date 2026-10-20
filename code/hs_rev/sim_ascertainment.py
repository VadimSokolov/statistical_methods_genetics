"""Ascertainment simulation of Section hs:sec:open (subsection "Ascertainment is
not a shrinkage problem", Example hs:ex:trunc and Table hs:tab:open).

The base simulation of Section hs:sec:sim (make_ftns_figures.make_design: L =
400 loci, 20 causal with dominance on 10, noise sd 0.10) is run with n = 400.
The horseshoe (ftns_lib.horseshoe, 1200 sweeps, 400 burn in) is fitted to
the full sample; then the lower half of the sample by measured fitness y is
discarded and the horseshoe is refitted to the 200 survivors (dosages and
fitness recentred within the survivors).  For each fit the posterior mean of
sigma^2_A = sum_l h_l alpha_l^2 is computed with
  - h from the full sample (full fit),
  - h from the survivors ("survivors", diagonal form, as in the text),
  - h from the full sample ("rescaled", the population genotypic variance),
  - the full survivor covariance V_S, b' V_S b ("survq", the first form of
    Proposition hs:prop:trunc; the diagonal forms above miss the negative
    covariances that truncation induces among the causal loci).
Each is divided by the true sigma^2_A of make_design.

Replicates: seeds 1 to 60 (sampler streams seed + 5000 and seed + 6000).  The
first twelve are also summarized separately.

Run:  python3 -I -B code/hs_rev/sim_ascertainment.py [nproc]   (about one minute on 6 cores)
Output: code/hs_rev/sim_ascertainment.txt and sim_ascertainment.json.
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

_mk = _os.makedirs
_os.makedirs = lambda *a, **k: None
import make_ftns_figures as M
_os.makedirs = _mk
import ftns_lib as F

N, NREP = 400, 60


def run(seed):
    D = M.make_design(seed, n=N)
    X, y, s2A = D["X"], D["y"], D["s2A"]
    h_full = X.var(0)
    B = F.horseshoe(X, y, n_iter=M.N_ITER, burn=M.BURN, rng=np.random.default_rng(seed + 5000))
    full = float(((B ** 2) @ h_full).mean())
    keep = y > np.median(y)
    Xs = X[keep] - X[keep].mean(0); ys = y[keep] - y[keep].mean()
    h_s = Xs.var(0)
    Bs = F.horseshoe(Xs, ys, n_iter=M.N_ITER, burn=M.BURN, rng=np.random.default_rng(seed + 6000))
    surv = float(((Bs ** 2) @ h_s).mean())
    resc = float(((Bs ** 2) @ h_full).mean())
    survq = float((((Xs @ Bs.T) ** 2).mean(0)).mean())     # full quadratic form b' V_S b, V_S survivor covariance
    R2 = s2A / float(np.var(y))
    return dict(seed=seed, s2A=s2A, R2=R2, nsurv=int(keep.sum()),
                full=full / s2A, surv=surv / s2A, resc=resc / s2A, survq=survq / s2A)


def block(rows, label):
    out = [label]
    for k in ("full", "surv", "resc", "survq", "R2"):
        v = np.array([r[k] for r in rows])
        out.append(f"  {k:5s} mean {v.mean():.4f}  se {v.std(ddof=1) / np.sqrt(len(v)):.4f}  "
                   f"min {v.min():.3f}  max {v.max():.3f}")
    s = np.array([r["surv"] for r in rows]); f = np.array([r["full"] for r in rows])
    out.append(f"  full/surv (mean of ratios) {np.mean(f / s):.2f};  ratio of means {f.mean() / s.mean():.2f}")
    from scipy.stats import norm
    d = (norm.pdf(0) / 0.5) ** 2; R2 = float(np.mean([r["R2"] for r in rows]))
    out.append(f"  large sample limits at mean R2 {R2:.3f} (Proposition hs:prop:trunc): "
               f"survivors, full form {(1 - d) ** 2 / (1 - d * R2):.3f}; population covariance {((1 - d) / (1 - d * R2)) ** 2:.3f}")
    return out


if __name__ == "__main__":
    nproc = int(_sys.argv[1]) if len(_sys.argv) > 1 else 6
    with Pool(nproc) as pool:
        R = pool.map(run, range(1, NREP + 1))
    json.dump(R, open(_HERE / "sim_ascertainment.json", "w"), indent=0)
    lines = [f"Ascertainment: n = {N}, L = 400, keep y > median (200 survivors); ratios to true sigma^2_A", ""]
    lines += block(R, f"all {NREP} replicates (seeds 1 to {NREP})")
    lines += [""] + block(R[:12], "first 12 replicates (seeds 1 to 12)")
    txt = "\n".join(lines)
    print(txt)
    open(_HERE / "sim_ascertainment.txt", "w").write(txt + "\n")
