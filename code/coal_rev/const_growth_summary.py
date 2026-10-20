"""Derived numbers quoted in Chapter coal from Tables coal:tab:const and coal:tab:growth and Section
coal:sec:calib, read from const_growth_seed<SEED>.json (written by const_growth.py).

Usage: python3 -I code/coal_rev/const_growth_summary.py [SEEDS]   (default 0,1; the chapter uses seed 0)
"""
import sys, json
from pathlib import Path as _Path
from scipy import stats
_HERE = _Path(__file__).resolve().parent

for seed in (sys.argv[1] if len(sys.argv) > 1 else "0,1").split(","):
    r = json.load(open(str(_HERE / ("const_growth_seed%s.json" % seed))))
    c, g = r["const"], r["growth"]; pt, pg = r["prior_width_theta"], r["prior_width_g"]
    print("seed", seed, " E[S] const %.1f growth %.1f  prior widths %.2f %.2f" % (r["ES_const_exact"], r["ES_growth"], pt, pg))
    for d, rows in (("const", ["xi theta", "S theta"]), ("growth", ["xi theta", "xi g", "S theta", "S g"])):
        for k in rows:
            x = r[d][k]; pw = pg if k.endswith(" g") else pt
            print("  %-6s %-9s width %7.2f cov90 %.3f cov50 %.3f SBC %6.1f  removes %.1f%% of prior width"
                  % (d, k, x["width"], x["cov90"], x["cov50"], x["sbc"], 100*(1-x["width"]/pw)))
    ratio = c["xi theta"]["width"]/c["S theta"]["width"]
    print("  const: spectrum interval narrower than S by %.1f%%" % (100*(1-ratio)))
    # coverage an S posterior interval would have if shrunk by this ratio (normal approximation)
    print("  normal approx coverage of S interval shrunk by that ratio: %.3f" % (2*stats.norm.cdf(1.6449*ratio)-1))
    print("  growth: theta improvement spectrum over S %.1f%%" % (100*(1-g["xi theta"]["width"]/g["S theta"]["width"])))
    print("  growth: g improvement spectrum over S %.1f%%" % (100*(1-g["xi g"]["width"]/g["S g"]["width"])))
    allcov = {(d, k): r[d][k] for d in ("const", "growth") for k in r[d] if isinstance(r[d][k], dict) and "prior" not in k}
    print("  max |cov90 - 0.90| %.4f, max |cov50 - 0.50| %.4f" % (max(abs(v["cov90"]-.9) for v in allcov.values()),
                                                               max(abs(v["cov50"]-.5) for v in allcov.values())))
    print("  five percent critical value chi2_9: %.2f" % stats.chi2.ppf(0.95, 9))
