"""Every number of Table coal:tab:reference (GBC rows), Section coal:sec:calib (SBC of the separately trained
S only network) and Section coal:sec:repair, from ref_gbc_seed*.json and ref_posterior_qmc.json.

Usage: python3 -I code/coal_rev/ref_gbc_summary.py   (output saved as ref_gbc_summary.txt)
"""
import json, numpy as np
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
P = [0.05, 0.25, 0.5, 0.75, 0.95]; S_LIST = [40, 100, 200]
ref = json.load(open(str(_HERE / "ref_posterior_qmc.json")))
seeds = sorted(int(p.stem.split("seed")[1]) for p in _HERE.glob("ref_gbc_seed*.json"))
for seed in seeds:
    r = json.load(open(str(_HERE / ("ref_gbc_seed%d.json" % seed))))
    print("=== training seed %d (chapter uses seed 0) ===" % seed)
    for p in ["theta", "g"]:
        b, a = r[p]["before"], r[p]["after"]
        print("%-5s before: width %.3f cov90 %.4f cov50 %.4f SBC %.1f thirds %s" % (p, b["width"], b["cov90"], b["cov50"], b["sbc"], np.round(b["cov90_thirds"], 3)))
        print("%-5s after : width %.3f cov90 %.4f cov50 %.4f SBC %.1f thirds %s  G^-1(.05,.95) %s" % (p, a["width"], a["cov90"], a["cov50"], a["sbc"], np.round(a["cov90_thirds"], 3), np.round(r[p]["Ginv_at_05_95"], 4)))
        R = np.array([ref[str(s)][p] for s in S_LIST]); Q = np.array([r[p]["gbc"][str(s)] for s in S_LIST])
        D = Q - R
        k = np.unravel_index(np.abs(D).argmax(), D.shape); kr = np.unravel_index(np.abs(D / R).argmax(), D.shape)
        for i, s in enumerate(S_LIST):
            print("   S=%3d ref %s gbc %s" % (s, np.round(R[i], 2), np.round(Q[i], 2)))
        print("   max abs disc %.3f at S=%d level %.2f (%.2f%% of quantile); max rel disc %.2f%% at S=%d level %.2f (abs %.3f)" % (
            abs(D[k]), S_LIST[k[0]], P[k[1]], 100 * abs(D[k] / R[k]), 100 * abs(D[kr] / R[kr]), S_LIST[kr[0]], P[kr[1]], abs(D[kr])))
print("reference standard errors (max over S and levels): theta %.4f g %.4f" % (
    max(max(ref[str(s)]["theta_se"]) for s in S_LIST), max(max(ref[str(s)]["g_se"]) for s in S_LIST)))
# Monte Carlo standard error of a 90% coverage within a third of the 2e4 test replicates, and the spread
# of the seed 0 theta coverages across thirds
r0 = json.load(open(str(_HERE / "ref_gbc_seed0.json")))
print("se of cov90 in a third: %.4f" % np.sqrt(0.9 * 0.1 / (20000 / 3)))
for k in ["before", "after"]:
    c = r0["theta"][k]["cov90_thirds"]; print("seed 0 theta %s: spread across thirds %.4f" % (k, max(c) - min(c)))
# null distribution of the SBC statistic after recalibration with an exact map: V uniform on the calibration
# and test sets (2e4 each), ranks of the test V among the 99 empirical calibration quantiles, ten bins
rng = np.random.default_rng(11); lev = np.arange(1, 100) / 100; chis = []
for _ in range(2000):
    Gi = np.quantile(rng.uniform(size=20000), lev); Vt = rng.uniform(size=20000)
    cnt = np.bincount(np.minimum(np.searchsorted(Gi, Vt) // 10, 9), minlength=10)
    chis.append(((cnt - 2000.0) ** 2 / 2000.0).sum())
chis = np.array(chis)
print("recalibrated exact map: SBC mean %.2f, P(SBC > 16.92) = %.3f, 95th percentile %.2f" % (chis.mean(), (chis > 16.92).mean(), np.quantile(chis, 0.95)))
