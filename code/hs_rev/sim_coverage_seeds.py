"""Direction of the coverage misses of Section hs:sec:open (Remark hs:rem:rerun).

The coverage figure ftns_coverage is one batch of sixty replicates (seeds 101
to 160) of make_ftns_figures.run_coverage.  This script reruns that experiment
on ten batches of sixty consecutive seeds, 101 to 700, with the same design,
sampler and sampler streams, for both the plain and the regularized horseshoe.
Batch 1 is the shipped figure run and reproduces code/ftns/summary.json.

For each prior it reports the coverage of the nominal 90 percent interval, the
number of intervals lying entirely above and entirely below the truth (pooled
and per batch), and how many batches have more misses above than below.

Run:  python3 -I -B code/hs_rev/sim_coverage_seeds.py [nproc]   (about 5 minutes on 6 cores)
Output: code/hs_rev/sim_coverage_seeds.txt and sim_coverage_seeds.json.
Nothing is written to code/ftns/ or figures/.
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
_os.makedirs = lambda *a, **k: None          # make_ftns_figures creates code/ftns/cache at import
import make_ftns_figures as M
_os.makedirs = _mk

NB, BS, SEED0 = 10, 60, 101


def summarize(rows):
    s2 = np.array([r["s2A"] for r in rows]); lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows])
    above = int(np.sum(lo > s2)); below = int(np.sum(hi < s2)); n = len(rows)
    return dict(n=n, cover=n - above - below, above=above, below=below,
                meanratio=float(np.mean([r["mean"] / r["s2A"] for r in rows])))


if __name__ == "__main__":
    nproc = int(_sys.argv[1]) if len(_sys.argv) > 1 else 6
    seeds = list(range(SEED0, SEED0 + NB * BS))
    tasks = [(s, reg) for reg in (False, True) for s in seeds]
    with Pool(nproc) as pool:
        out = pool.map(M.run_coverage, tasks, chunksize=4)
    json.dump(out, open(_HERE / "sim_coverage_seeds.json", "w"))
    lines = [f"Coverage experiment, {NB} batches of {BS} replicates, seeds {SEED0} to {SEED0 + NB * BS - 1}", ""]
    res = {}
    for reg, tag in ((False, "plain"), (True, "regularized")):
        rows = [r for r in out if r["reg"] == reg]
        P = summarize(rows)
        n, a, b, c = P["n"], P["above"], P["below"], P["cover"]
        miss = a + b
        lines.append(f"{tag}: coverage {c}/{n} = {c / n:.3f} (se {np.sqrt(c / n * (1 - c / n) / n):.3f}); "
                     f"misses {miss}: above {a} ({a / n:.3f}), below {b} ({b / n:.3f}); "
                     f"share of misses above {a / miss:.3f} (se {np.sqrt(a / miss * (1 - a / miss) / miss):.3f}); "
                     f"mean ratio {P['meanratio']:.4f}")
        nab = nbel = ntie = 0
        for k in range(NB):
            bs = set(range(SEED0 + k * BS, SEED0 + (k + 1) * BS))
            B = summarize([r for r in rows if r["seed"] in bs])
            nab += B["above"] > B["below"]; nbel += B["above"] < B["below"]; ntie += B["above"] == B["below"]
            lines.append(f"   batch {k + 1:2d} seeds {min(bs)}-{max(bs)}: coverage {B['cover'] / BS:.3f}, "
                         f"above {B['above']:2d}, below {B['below']:2d}, mean ratio {B['meanratio']:.3f}")
        lines.append(f"   batches with more misses above: {nab}, more below: {nbel}, tied: {ntie}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    open(_HERE / "sim_coverage_seeds.txt", "w").write(txt + "\n")
