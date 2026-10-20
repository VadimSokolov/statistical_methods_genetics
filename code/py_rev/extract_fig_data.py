"""Recover the plotted data of five py figures from their original matplotlib PDFs.

The figures py_fig-counting, py_fig-coincidence, py_fig-diversity, py_fig-estimation and
py_fig-index were drawn by scripts that are not in the repository. This script reads the
vector paths of the original PDFs (kept in misc/review_r2/baseline_figures/), maps them to data
coordinates through the major tick marks, and writes fig_data.json beside it. The redraw scripts
fig_<name>.py read that file. py_fig-exponent needs no recovery: it is a closed form.

Run once: python3 -I code/py_rev/extract_fig_data.py   (needs PyMuPDF)
"""
import json
import math
from pathlib import Path as _Path

import fitz

_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
BASE = _ROOT / "misc" / "review_r2" / "baseline_figures"

BLUE, RED, GREEN, PURPLE = (0.122, 0.467, 0.706), (0.839, 0.153, 0.157), (0.173, 0.627, 0.173), (0.58, 0.404, 0.741)


def col(c):
    return None if c is None else tuple(round(v, 3) for v in c)


def drawings(name):
    return fitz.open(str(BASE / f"py_fig-{name}.pdf"))[0].get_drawings()


def lines(name):
    """Coloured polylines with their points in PDF coordinates (legend handles included)."""
    out = []
    for d in drawings(name):
        it = d["items"]
        if d.get("fill") is None and it and all(i[0] == "l" for i in it):
            pts = [(it[0][1].x, it[0][1].y)] + [(i[2].x, i[2].y) for i in it]
            out.append((col(d.get("color")), pts))
    return out


def axis_map(p0, v0, p1, v1, log=False):
    """Linear map from PDF coordinate to data value, through two ticks."""
    f = (lambda v: math.log10(v)) if log else (lambda v: v)
    g = (lambda u: 10 ** u) if log else (lambda u: u)
    a, b = f(v0), f(v1)
    return lambda p: g(a + (p - p0) * (b - a) / (p1 - p0))


def series(name, color, xr, fx, fy, npts):
    """The polyline of the given colour inside the x range with npts points."""
    for c, pts in lines(name):
        if c == color and len(pts) == npts and xr[0] <= pts[0][0] <= xr[1]:
            return [fx(x) for x, _ in pts], [fy(y) for _, y in pts]
    raise RuntimeError((name, color))


def r(v, k=6):
    return [float(f"{x:.{k}g}") for x in v]


data = {}

# counting: x = 1/x on log axis (ticks 1e5 .. 1e9), y = E[N(x)] log (ticks 1e2 .. 1e5)
fx = axis_map(69.881, 1e5, 431.899, 1e9, log=True)
fy = axis_map(200.086, 1e2, 47.294, 1e5, log=True)
cnt = {}
for key, c in [("s0.5_a1", BLUE), ("s0.5_a2", RED), ("s0.5_a0.5", GREEN), ("s0.3_a2", PURPLE)]:
    x, y = series("counting", c, (60, 440), fx, fy, 9)
    cnt[key] = {"x": r(x[::-1], 4), "y": r(y[::-1])}
data["counting"] = cnt

# coincidence: left n log (1e3 .. 1e7), E[A1] log (1e0 .. 1e3); right Pr linear (0 .. 1)
fxl = axis_map(65.271, 1e3, 328.512, 1e7, log=True)
fyl = axis_map(218.419, 1.0, 69.404, 1e3, log=True)
fxr = axis_map(403.996, 1e3, 667.238, 1e7, log=True)
fyr = axis_map(224.448, 0.0, 30.472, 1.0)
coin = {"n": None, "A1": {}, "distinct": {}}
for key, c in [("0", BLUE), ("0.2", RED), ("0.5", GREEN), ("0.7", PURPLE)]:
    x, y = series("coincidence", c, (60, 340), fxl, fyl, 5)
    coin["n"] = r(x, 4)
    coin["A1"][key] = r(y)
    x, y = series("coincidence", c, (400, 680), fxr, fyr, 5)
    coin["distinct"][key] = [max(0.0, v) for v in r(y, 4)]
data["coincidence"] = coin

# estimation: alpha on a base 2 log axis (ticks 0.25 .. 4); left y 0.2 .. 0.8, right y 0 .. 1
fxl = axis_map(61.325, 0.25, 333.435, 4.0, log=True)
fyl = axis_map(228.088, 0.2, 34.42, 0.8)
fxr = axis_map(409.085, 0.25, 681.195, 4.0, log=True)
fyr = axis_map(238.12, 0.0, 24.72, 1.0)
est = {"alpha": [0.25, 0.5, 1.0, 2.0, 4.0], "left": {}, "right": {}}
for key, c in [("logK", BLUE), ("spec2", RED), ("A1K", GREEN)]:
    x, y = series("estimation", c, (55, 340), fxl, fyl, 5)
    est["left"][key] = r(y, 4)
    x, y = series("estimation", c, (400, 690), fxr, fyr, 5)
    est["right"][key] = r(y, 4)
x, y = series("estimation", (0.0, 0.0, 0.0), (55, 340), fxl, fyl, 5)
est["left"]["truth"] = r(y, 4)
data["estimation"] = est

# index: n log (1e3 .. 1e8), A1/Kn linear (0.2 .. 0.6)
fx = axis_map(74.323, 1e3, 445.825, 1e8, log=True)
fy = axis_map(242.728, 0.2, 26.16, 0.6)
idx = {"n": None}
for key, c in [("pure", BLUE), ("log", RED), ("poly", GREEN)]:
    x, y = series("index", c, (60, 460), fx, fy, 6)
    idx["n"] = r(x, 4)
    idx[key] = r(y, 4)
data["index"] = idx

# diversity: left scatter S, S*; curve C S^kappa; right histograms (density)
fxl = axis_map(86.413, 1.0, 312.371, 6.0)
fyl = axis_map(223.064, 0.0, 36.738, 10.0)
fxr = axis_map(395.627, 0.0, 627.219, 8.0)
fyr = axis_map(228.76, 0.0, 66.505, 0.4)
pts, bars = [], {"S": [], "Sstar": []}
for d in drawings("diversity"):
    f, it, rc = col(d.get("fill")), d["items"], d["rect"]
    if f == BLUE and abs((d.get("fill_opacity") or 1) - 0.6) < 0.01:
        cx, cy = (rc.x0 + rc.x1) / 2, (rc.y0 + rc.y1) / 2
        if not (cx < 80 and cy < 60):  # drop the legend marker at the top left
            pts.append((fxl(cx), fyl(cy)))
    elif f in (BLUE, RED) and abs((d.get("fill_opacity") or 1) - 0.55) < 0.01 and rc.x0 > 387:
        if rc.y1 < 60:  # legend patch
            continue
        bars["S" if f == BLUE else "Sstar"].append((fxr(rc.x0), fxr(rc.x1), fyr(rc.y0)))
x, y = series("diversity", RED, (47, 341), fxl, fyl, 24)
for k in bars:
    bars[k].sort()
data["diversity"] = {
    "S": r([p[0] for p in pts]), "Sstar": r([p[1] for p in pts]),
    "curve_S": r(x), "curve_Sstar": r(y),
    "hist_S": [r(b, 5) for b in bars["S"]], "hist_Sstar": [r(b, 5) for b in bars["Sstar"]],
}

(_HERE / "fig_data.json").write_text(json.dumps(data, indent=1))
for k, v in data.items():
    print(k, {kk: (len(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()} if k == "diversity" else v)
