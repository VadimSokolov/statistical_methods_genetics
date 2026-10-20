"""Numbers quoted in Section hs:sec:sim, Table hs:tab:sim, the chapter summary,
the coverage subsection and Example hs:ex:trunc of chapter hs.

The six figures ftns_variance, ftns_rate, ftns_shrinkage, ftns_scaling,
ftns_coverage and ftns_secondary, and code/ftns/summary.json, come from one run
of code/ftns/make_ftns_figures.py.  This script

  1. reads code/ftns/summary.json and prints every quantity the text quotes,
     rounded as in the text;
  2. reruns the twelve replicates of the main experiment (seeds 1 to 12, the
     same seeds and sampler streams as make_ftns_figures.run_main), checks that
     the rerun reproduces summary.json, and adds three quantities that
     summary.json does not store: the mean phenotypic variance var(y), the
     predicted marginal bias L var(y)/n with its Monte Carlo standard error,
     and the coverage of the nominal 90 percent posterior interval for
     sigma^2_A in the twelve replicates.

Run:  python3 -I code/hs_rev/sim_main_check.py        (about one minute)
Output is cached in code/hs_rev/sim_main_check.txt.
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
import json
import numpy as np

# make_ftns_figures creates code/ftns/cache at import time; suppress that.
_mk = _os.makedirs
_os.makedirs = lambda *a, **k: None
import make_ftns_figures as M
_os.makedirs = _mk

S = json.load(open(_FTNS / "summary.json"))

print("== summary.json, as quoted in the text ==")
print(f"truth s2A {S['truth_s2A']:.4f}  MC se {S['truth_s2A_se']:.4f}  s2D {S['s2D']:.4f}")
print(f"realized {S['dw_real']:.6f}  partial {S['dw_partial']:.6f}  ratio of means {S['dw_real']/S['dw_partial']:.5f}"
      f"  mean of ratios {S['ratio_real_partial']:.5f}")
for k in ["marg", "ridge", "hs_plug", "hs_post"]:
    print(f"{k:8s} mean {S[k]:.4f}  ratio {S[k + '_ratio']:.2f}")
print(f"observed marginal bias {S['marg_bias']:.4f}")
print(f"MSE marg {S['mse_marg']:.3g} ridge {S['mse_ridge']:.3g} hs {S['mse_hs']:.3g};"
      f" folds marg/hs {S['mse_marg']/S['mse_hs']:.1f} ridge/hs {S['mse_ridge']/S['mse_hs']:.1f}")
print(f"kappa all reps: null {S['kappa_null']:.2f} signal {S['kappa_signal']:.2f};"
      f" replicate 1 (figure): null {S['kappa_rep1_null']:.2f} signal {S['kappa_rep1_signal']:.2f}")
for L in (100, 200, 400, 800):
    print(f"scaling L={L}: marg {S[f'scale_marg_{L}']:.2f} (minus one {S[f'scale_marg_{L}']-1:.2f})"
          f"  hs {S[f'scale_hs_{L}']:.2f}")
e = [S[f"scale_marg_{L}"] - 1 for L in (100, 200, 400, 800)]
print("successive ratios of (ratio minus one):", " ".join(f"{b/a:.2f}" for a, b in zip(e, e[1:])))
for t in ("plain", "reg"):
    print(f"coverage {t}: {100*S['cov_' + t]:.1f}%  below {100*S['below_' + t]:.1f}%  above {100*S['above_' + t]:.1f}%"
          f"  se {100*np.sqrt(S['cov_' + t]*(1-S['cov_' + t])/60):.1f}%  misses {round(60*(1-S['cov_' + t]))}"
          f"  mean ratio {S['meanratio_' + t]:.3f}")
print(f"secondary: marg {S['sec_marg']:.2f} hs plug {S['sec_hs_plug']:.2f} hs post {S['sec_hs_post']:.2f}")

print("\n== rerun of the twelve main replicates ==")
L, n = 400, 300
rows = []
for seed in range(1, 13):
    D = M.make_design(seed)
    X, y, h = D["X"], D["y"], D["h"]
    am = M.marginal(X, y)
    B, kap = M.hs_kappa(X, y, rng=np.random.default_rng(seed + 1000))
    s = (B ** 2) @ h
    lo, hi = np.quantile(s, [0.05, 0.95])
    rows.append(dict(s2A=D["s2A"], vy=float(np.var(y)), marg=float(h @ am ** 2),
                     hs_post=float(s.mean()), lo=float(lo), hi=float(hi)))
A = lambda k: np.array([r[k] for r in rows])
tr = A("s2A")
pred = L * A("vy") / n
obs = A("marg") - tr
print(f"rerun truth {tr.mean():.5f} (summary {S['truth_s2A']:.5f}); marg ratio {(A('marg')/tr).mean():.3f}"
      f" (summary {S['marg_ratio']:.3f}); hs post ratio {(A('hs_post')/tr).mean():.3f} (summary {S['hs_post_ratio']:.3f})")
print(f"mean var(y) {A('vy').mean():.4f}; R2 = s2A/var(y) {tr.mean()/A('vy').mean():.3f}")
print(f"predicted bias L var(y)/n {pred.mean():.4f}; observed bias {obs.mean():.4f};"
      f" difference {(obs - pred).mean():.4f}, MC se {(obs - pred).std(ddof=1)/np.sqrt(12):.4f}")
print(f"predicted bias with the error variance 0.01 only: {L*0.01/n:.4f}")
hit = (A("lo") <= tr) & (tr <= A("hi"))
print(f"90% interval coverage in the 12 main replicates: {hit.sum()}/12 = {100*hit.mean():.1f}%"
      f" (se {100*np.sqrt(hit.mean()*(1-hit.mean())/12):.1f}%); below {(A('hi') < tr).sum()}, above {(A('lo') > tr).sum()}")

print("\n== rerun of the scaling experiment (seeds 201 to 206 per L) ==")
for L in (100, 200, 400, 800):
    rs = [M.run_scaling((s, L)) for s in range(201, 207)]
    for k in ("marg", "hs_post"):
        v = np.array([r[k] / r["s2A"] for r in rs])
        print(f"L={L} {k:8s} ratio {v.mean():.3f}  MC se {v.std(ddof=1)/np.sqrt(len(v)):.3f}"
              f"  (ratio minus one)/se {abs(v.mean()-1)/(v.std(ddof=1)/np.sqrt(len(v))):.2f}")
