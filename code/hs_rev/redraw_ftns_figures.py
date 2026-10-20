"""Redraw the six figures of chapter hs (ftns_variance, ftns_rate,
ftns_shrinkage, ftns_scaling, ftns_coverage, ftns_secondary) to house style.

The data are the same run as code/ftns/make_ftns_figures.py and
code/ftns/summary.json: main experiment seeds 1 to 12, scaling seeds 201 to
206 per L, coverage seeds 101 to 160, secondary seeds 301 to 312, with the same
sampler streams.  The workers are imported from make_ftns_figures and rerun
(about three minutes on 6 cores; results cached in code/hs_rev/cache_figs/).

Differences from make_ftns_figures: every figure is saved at its exact
figsize (no tight bounding box), 5.4 in (full) or 3.8 in (narrow) wide, so
that the chapter includes it at scale one with a width of 0.70 or 1.00 of the
text block; every text element is 9 pt; and the mathtext script ratio is
raised from 0.70 to 0.78 so that sub and superscripts print at 7 pt.

Run:  SMG_FIG_DIR=/tmp/smg_fig python3 -I -B code/hs_rev/redraw_ftns_figures.py [nproc]
(writes to figures/ when SMG_FIG_DIR is unset).
"""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FTNS = _ROOT / "code" / "ftns"
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib._mathtext as _mt

_mt.SHRINK_FACTOR = 0.78          # 9 pt x 0.78 = 7 pt for sub and superscripts
CACHE = _HERE / "cache_figs"
FS = 9
C = M.C
NARROW, WIDE = 3.8, 5.4


def style():
    M.style()
    plt.rcParams.update({"axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                         "legend.fontsize": FS, "axes.titlesize": FS, "font.size": FS,
                         "savefig.bbox": None, "savefig.pad_inches": 0.0})


def save(fig, name):
    fig.savefig(_FIGDIR / f"{name}.pdf")
    plt.close(fig)


def cached(name, fn, tasks, pool):
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.load(open(path))
    out = pool.map(fn, tasks)
    CACHE.mkdir(exist_ok=True)
    json.dump(out, open(path, "w"))
    return out


def fig_variance(R):
    names = ["marg", "ridge", "hs_plug", "hs_post"]
    labs = ["marginal", "ridge (CV)", "horseshoe plug in", "horseshoe posterior"]
    cols = [C["marg"], C["ridge"], C["hsplug"], C["hspost"]]
    truth = np.array([r["s2A"] for r in R])
    fig, ax = plt.subplots(1, 2, figsize=(WIDE, 3.2))
    for i, (k, c) in enumerate(zip(names, cols)):
        v = np.array([r[k] for r in R]); x = M.jitter(len(v), i)
        ax[0].scatter(x, v, s=10, color=c, edgecolor="none", alpha=0.9, zorder=3)
        ax[0].hlines(v.mean(), i - 0.25, i + 0.25, color=C["black"], lw=0.9, zorder=4)
        ax[1].scatter(x, v / truth, s=10, color=c, edgecolor="none", alpha=0.9, zorder=3)
        ax[1].hlines((v / truth).mean(), i - 0.25, i + 0.25, color=C["black"], lw=0.9, zorder=4)
    ax[0].scatter(M.jitter(len(truth), 4), truth, s=10, color=C["grey"], edgecolor="none", zorder=3)
    ax[0].hlines(truth.mean(), 3.75, 4.25, color=C["black"], lw=0.9)
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels(labs + ["truth"], rotation=35, ha="right", rotation_mode="anchor")
    ax[0].set_ylabel(r"estimated $\sigma^2_A$")
    ax[1].axhline(1.0, color=C["grey"], ls=":", lw=1.0)
    ax[1].set_yscale("log")
    ax[1].set_yticks([0.25, 0.5, 1, 2, 4]); ax[1].set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    ax[1].minorticks_off()
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labs, rotation=35, ha="right", rotation_mode="anchor")
    ax[1].set_ylabel(r"$\hat\sigma^2_A/\sigma^2_A$")
    for a in ax:
        a.set_xlim(-0.5, a.get_xticks()[-1] + 0.5)
        a.tick_params(axis="x", length=0)
    fig.tight_layout(w_pad=1.0, pad=0.3)
    save(fig, "ftns_variance")


def fig_rate(R):
    real = np.array([r["dw_real"] for r in R]); wb = np.array([r["wbar"] for r in R])
    fig, ax = plt.subplots(figsize=(NARROW, 3.0))
    for k, c, lab, mk in [("marg", C["marg"], "marginal", "o"), ("ridge", C["ridge"], "ridge (CV)", "s"),
                          ("hs_post", C["hspost"], "horseshoe posterior", "D")]:
        v = np.array([r[k] for r in R]) / wb
        ax.scatter(real, v, s=13, marker=mk, color=c, edgecolor="none", label=lab, zorder=3)
    lim = [0.004, 0.15]
    ax.plot(lim, lim, color=C["grey"], ls="--", lw=0.9, label="equality")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.008, 0.04); ax.set_ylim(*lim)
    ax.set_xlabel(r"realized $\Delta\bar w$")
    ax.set_ylabel(r"predicted $\hat\sigma^2_A/\bar w$")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3)
    fig.tight_layout(pad=0.3)
    save(fig, "ftns_rate")


def fig_shrinkage(r):
    kap = np.array(r["kappa"]); am = np.array(r["absmarg"]); cz = np.array(r["causal"])
    fig, ax = plt.subplots(figsize=(NARROW, 2.8))
    ax.scatter(am[~cz], kap[~cz], s=6, color=C["grey"], alpha=0.6, edgecolor="none",
               label=f"null loci (mean {kap[~cz].mean():.2f})")
    ax.scatter(am[cz], kap[cz], s=16, color=C["hspost"], edgecolor="none",
               label=f"nonzero effect (mean {kap[cz].mean():.2f})")
    ax.set_xlabel(r"$|\hat\alpha^{\,\mathrm{marg}}_\ell|$")
    ax.set_ylabel(r"posterior mean of $\kappa_\ell$")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.2)
    fig.tight_layout(pad=0.3)
    save(fig, "ftns_shrinkage")


def fig_scaling(S, Ls):
    fig, ax = plt.subplots(figsize=(NARROW, 2.8))
    for k, c, lab, mk, off in [("marg", C["marg"], "marginal plug in", "o", -0.02),
                               ("hs_post", C["hspost"], "horseshoe posterior", "D", 0.02)]:
        mu, se = [], []
        for L in Ls:
            v = np.array([s[k] / s["s2A"] for s in S if s["L"] == L])
            mu.append(v.mean()); se.append(v.std(ddof=1) / np.sqrt(len(v)))
        mu, se = np.array(mu), np.array(se)
        ax.errorbar(np.array(Ls) * (1 + off), mu, yerr=2 * se, color=c, marker=mk, ms=4, capsize=2,
                    lw=1.0, elinewidth=0.8, label=lab)
    ax.axhline(1.0, color=C["grey"], ls=":", lw=1.0)
    ax.set_xscale("log"); ax.set_xticks(Ls); ax.set_xticklabels([str(L) for L in Ls]); ax.minorticks_off()
    ax.set_xlabel(r"number of loci $L$")
    ax.set_ylabel(r"$\hat\sigma^2_A/\sigma^2_A$")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    save(fig, "ftns_scaling")


def fig_coverage(Cv):
    fig, ax = plt.subplots(1, 2, figsize=(WIDE, 2.8), sharey=True)
    for a, reg, title, col in [(ax[0], False, "horseshoe", C["hspost"]),
                               (ax[1], True, "regularized horseshoe", C["reg"])]:
        rr = sorted([c for c in Cv if c["reg"] == reg], key=lambda c: c["s2A"])
        cov = 0
        for i, c in enumerate(rr):
            hit = c["lo"] <= c["s2A"] <= c["hi"]; cov += hit
            a.vlines(i, c["lo"], c["hi"], color=col if hit else C["black"], lw=0.9 if hit else 1.3,
                     alpha=0.75 if hit else 1.0)
            a.hlines(c["s2A"], i - 0.45, i + 0.45, color=C["marg"], lw=1.0)
        a.set_title(f"{title}: coverage {100 * cov / len(rr):.1f}%", fontsize=FS)
        a.set_xlabel("replicate, ordered by true $\\sigma^2_A$")
        a.set_yscale("log"); a.set_xticks([])
    ax[0].set_ylabel(r"90% interval for $\sigma^2_A$")
    fig.tight_layout(w_pad=0.8, pad=0.3)
    save(fig, "ftns_coverage")


def fig_secondary(Sx):
    truth = np.array([s["truth"] for s in Sx])
    names = ["marg", "hs_plug", "hs_post"]
    labs = ["marginal\nplug in", "bivariate\nhorseshoe\nplug in", "bivariate\nhorseshoe\nposterior"]
    cols = [C["marg"], C["hsplug"], C["hspost"]]
    fig, ax = plt.subplots(figsize=(NARROW, 2.9))
    for i, (k, c) in enumerate(zip(names, cols)):
        v = np.array([s[k] for s in Sx]) / truth
        ax.boxplot(v, positions=[i], widths=0.45, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor=C["grey"], lw=0.7),
                   medianprops=dict(color=C["black"], lw=0.9), whiskerprops=dict(color=C["grey"], lw=0.7),
                   capprops=dict(color=C["grey"], lw=0.7))
        ax.scatter(M.jitter(len(v), i, 0.1), v, s=11, color=c, edgecolor="none", zorder=3)
    ax.axhline(1.0, color=C["grey"], ls=":", lw=1.0)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5", "1", "2", "4"]); ax.minorticks_off()
    ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("estimate / true covariance")
    fig.tight_layout(pad=0.3)
    save(fig, "ftns_secondary")


if __name__ == "__main__":
    nproc = int(_sys.argv[1]) if len(_sys.argv) > 1 else 6
    Ls = [100, 200, 400, 800]
    with Pool(nproc) as pool:
        R = cached("main", M.run_main, list(range(1, 13)), pool)
        S = cached("scaling", M.run_scaling, [(s, L) for L in Ls for s in range(201, 207)], pool)
        Cv = cached("coverage", M.run_coverage, [(s, reg) for reg in (False, True) for s in range(101, 161)], pool)
        Sx = cached("secondary", M.run_secondary, list(range(301, 313)), pool)
    # check against summary.json
    Sm = json.load(open(_FTNS / "summary.json"))
    chk = {"hs_post_ratio": np.mean([r["hs_post"] / r["s2A"] for r in R]),
           "cov_plain": np.mean([c["lo"] <= c["s2A"] <= c["hi"] for c in Cv if not c["reg"]]),
           "sec_hs_post": np.mean([s["hs_post"] / s["truth"] for s in Sx]),
           "scale_marg_800": np.mean([s["marg"] / s["s2A"] for s in S if s["L"] == 800])}
    for k, v in chk.items():
        print(f"{k:16s} rerun {v:.6f}  summary.json {Sm[k]:.6f}")
    _FIGDIR.mkdir(parents=True, exist_ok=True)
    style()
    fig_variance(R); fig_rate(R); fig_shrinkage(R[0]); fig_scaling(S, Ls); fig_coverage(Cv); fig_secondary(Sx)
    print("wrote six figures to", _FIGDIR)
