"""Recover the plotted data of ftns figures f2, f3, f5 and f6 from their original PDFs.

These four figures had no script in the repository. The originals were matplotlib PDFs, so
the plotted polylines survive as vector paths. This script reads the saved originals in
misc/review_r2/baseline_figures/, maps page coordinates to data coordinates with the tick
marks of each axis (linear or log scale), and writes ftns_figdata.json beside itself, which
the fig_ftns_f*.py scripts read. Run once; it only reads the PDFs.

Calibration ticks (page points, value) were read from the tick marks and tick labels of each
original. Abscissae on a known grid (generations 0 to 15, the recombination rates, the
sample sizes) are snapped to that grid. One point is reconstructed rather than read: the
d = 0.20 curve of f6 leaves the page at n = 2000, where matplotlib clipped the segment; its
value is extrapolated along the clipped segment, which lies far below the plotted range.

Requires PyMuPDF (fitz). Run with python3 -I.
"""
import os as _os, sys as _sys, json
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import numpy as np
import fitz

BASE = _ROOT / "misc" / "review_r2" / "baseline_figures"
OUT = _HERE / "ftns_figdata.json"


def drawings(name):
    page = fitz.open(str(BASE / f"{name}.pdf"))[0]
    out = []
    for g in page.get_drawings():
        pts = []
        for it in g["items"]:
            if it[0] == "l":
                pts += [it[1], it[2]]
            elif it[0] == "c":
                pts += [it[1], it[4]]
        P = []
        for q in pts:
            if not P or abs(P[-1][0] - q.x) + abs(P[-1][1] - q.y) > 1e-6:
                P.append((q.x, q.y))
        col = tuple(round(c, 3) for c in g["color"]) if g.get("color") else None
        out.append(dict(type=g["type"], color=col, pts=np.array(P), rect=g["rect"],
                        width=g.get("width")))
    return out


def lin(p0, v0, p1, v1):
    return lambda p: v0 + (np.asarray(p) - p0) * (v1 - v0) / (p1 - p0)


def log(p0, v0, p1, v1):
    f = lin(p0, np.log10(v0), p1, np.log10(v1))
    return lambda p: 10.0 ** f(p)


def lines(D, color, xlo, xhi, minpts=3):
    """Stroked polylines of a colour whose x range lies inside [xlo, xhi], legend lines excluded."""
    res = []
    for g in D:
        if g["type"] != "s" or g["color"] != color or len(g["pts"]) < minpts:
            continue
        x = g["pts"][:, 0]
        if x.min() >= xlo and x.max() <= xhi and x.max() - x.min() > 40:
            res.append(g["pts"])
    return res


BLUE = (0.122, 0.306, 0.475)
RED = (0.753, 0.224, 0.169)
GREY = (0.541, 0.604, 0.647)
PURPLE = (0.424, 0.204, 0.514)
AMBER = (0.839, 0.537, 0.063)
data = {}

# f2: exclusive or system, two linear panels, generations 0..15
D = drawings("ftns_f2_xor")
gx_l = lin(58.148, 0.0, 240.393, 15.0); gy_l = lin(149.904, 0.85, 25.450, 1.10)
gx_r = lin(302.923, 0.0, 485.168, 15.0); gy_r = lin(144.254, -0.20, 30.926, 0.0)
w = lines(D, BLUE, 49, 250)[0]
fp = lines(D, RED, 49, 250)
fp = [p for p in fp if np.ptp(p[:, 1]) < 1e-6][0]
dd = lines(D, RED, 293, 495)[0]
pa = lines(D, GREY, 293, 495)[0]
data["f2"] = dict(gen=np.rint(gx_l(w[:, 0])).astype(int).tolist(),
                  wbar=np.round(gy_l(w[:, 1]), 5).tolist(),
                  fisher=float(np.round(gy_l(fp[0, 1]), 5)),
                  D=np.round(gy_r(dd[:, 1]), 5).tolist(),
                  pA_minus_half=np.round(gy_r(pa[:, 1]), 5).tolist(),
                  check_gen_right=np.rint(gx_r(dd[:, 0])).astype(int).tolist())

# f3: order of the remainder, log log panels
D = drawings("ftns_f3_order")
sx_l = log(81.38, 1e-3, 203.16, 1e-1); ry = log(155.86, 1e-4, 32.69, 1e-1)
sx_r = log(327.12, 1e-3, 448.91, 1e-1); ny = log(141.15, 1e4, 38.24, 1e14)
rem = lines(D, BLUE, 54, 249)[0]
lin28 = lines(D, RED, 54, 249)[0]
nst = lines(D, PURPLE, 300, 495)[0]
n50 = lines(D, RED, 300, 495)[0]
data["f3"] = dict(s=sx_l(rem[:, 0]).tolist(), rel_remainder=ry(rem[:, 1]).tolist(),
                  line028=ry(lin28[:, 1]).tolist(),
                  s_right=sx_r(nst[:, 0]).tolist(), nstar=ny(nst[:, 1]).tolist(),
                  line50=ny(n50[:, 1]).tolist(), assay=float(ny(141.15)),
                  xlim=[float(sx_l(54.28)), float(sx_l(248.5))],
                  ylim_left=[float(ry(155.86)), float(ry(22.9))],
                  ylim_right=[float(ny(155.86)), float(ny(22.9))])

# f5: quasi linkage equilibrium, r on a log axis
D = drawings("ftns_f5_qle")
rx_l = log(114.66, 0.1, 169.41, 0.2); dy = log(132.42, 1e-3, 47.24, 1e-2)
rx_r = log(358.05, 0.1, 412.80, 0.2); ey = lin(155.86, 0.0, 38.33, 0.4)
RGRID = np.array([0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])


def snap(v):
    i = np.argmin(np.abs(np.log(RGRID)[None, :] - np.log(v)[:, None]), axis=1)
    assert np.all(np.abs(np.log(RGRID[i] / v)) < 0.02), v
    return RGRID[i]


f5 = {}
for key, col in [("0.005", BLUE), ("0.01", RED), ("0.02", PURPLE)]:
    L = lines(D, col, 50, 251)[0]; R = lines(D, col, 294, 495)[0]
    oL = np.argsort(L[:, 0]); oR = np.argsort(R[:, 0])
    f5[key] = dict(r=snap(rx_l(L[oL, 0])).tolist(), Dstar=dy(L[oL, 1]).tolist(),
                   r_right=snap(rx_r(R[oR, 0])).tolist(), relerr=ey(R[oR, 1]).tolist())
th = lines(D, (0.0, 0.0, 0.0), 50, 251)
th = [p for p in th if len(p) > 3][0]; o = np.argsort(th[:, 0])
f5["theory"] = dict(r=snap(rx_l(th[o, 0])).tolist(), Dstar=dy(th[o, 1]).tolist())
f5["flat"] = float(ey(82.4))
f5["xlim"] = [float(rx_l(50.81)), float(rx_l(250.9))]
f5["ylim_left"] = [float(dy(155.86)), float(dy(23.6))]
f5["ylim_right"] = [0.0, float(ey(23.6))]
data["f5"] = f5

# f6: Bayes factor against n, log log
D = drawings("ftns_f6_bf")
nx = log(111.00, 1e2, 323.14, 1e5); ky = log(28.01, 1e2, 190.78, 1e-30)
NGRID = np.array([25, 50, 100, 200, 400, 1000, 2000, 4000, 10000, 20000, 50000, 100000])


def nsnap(v):
    i = np.argmin(np.abs(np.log(NGRID)[None, :] - np.log(v)[:, None]), axis=1)
    assert np.all(np.abs(np.log(NGRID[i] / v)) < 0.02), v
    return NGRID[i]


f6 = {}
for key, col in [("d020", RED), ("d002", AMBER), ("d0", BLUE), ("sqrtn", GREY)]:
    P = lines(D, col, 55, 336, minpts=3)
    P = max(P, key=len)
    if key == "d020":
        # last vertex is the clip point at the page edge: extend along the segment to n = 2000
        (x0, y0), (x1, y1) = P[-2], P[-1]
        xt = 111.00 + (323.14 - 111.00) * (np.log10(2000) - 2) / 3
        yt = y0 + (y1 - y0) * (xt - x0) / (x1 - x0)
        P = np.vstack([P[:-1], [xt, yt]])
    f6[key] = dict(n=nsnap(nx(P[:, 0])).tolist(), K=ky(P[:, 1]).tolist())
f6["xlim"] = [float(nx(55.7)), float(nx(335.9))]
f6["ylim"] = [float(ky(190.78)), float(ky(22.9))]
f6["hline"] = float(ky(38.2))
data["f6"] = f6

with open(OUT, "w") as fh:
    json.dump(data, fh, indent=1)
print("wrote", OUT)
print("f2 wbar", data["f2"]["wbar"][:3], "...", data["f2"]["wbar"][-1], "fisher", data["f2"]["fisher"])
print("f2 D", data["f2"]["D"][0], data["f2"]["D"][-1], "pA", set(data["f2"]["pA_minus_half"]))
print("f3 s range", data["f3"]["s"][0], data["f3"]["s"][-1], len(data["f3"]["s"]))
print("f3 ratio rem/s", np.array(data["f3"]["rel_remainder"])[[0, -1]] / np.array(data["f3"]["s"])[[0, -1]])
print("f3 nstar*s^4", np.array(data["f3"]["nstar"])[[0, -1]] * np.array(data["f3"]["s_right"])[[0, -1]] ** 4)
print("f5 relerr eps .01", np.round(f5["0.01"]["relerr"], 3), "flat", f5["flat"])
print("f6 d0 K", np.round(f6["d0"]["K"], 2), f6["d0"]["n"])
print("f6 d020", f6["d020"]["n"], ["%.2g" % k for k in f6["d020"]["K"]])
