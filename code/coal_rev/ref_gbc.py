"""GBC rows of Table coal:tab:reference, the SBC statistics of the separately trained S only network cited in
Section coal:sec:calib, and the rank recalibration numbers of Section coal:sec:repair (Chapter coal).

Round 3 recomputation (the original network was lost). Design of Section coal:sec:reference: n = 30,
theta ~ Ga(4, scale 25), g ~ Unif(0, 20), N(t) = N0 exp(-g t) backwards in time, S | L ~ Poisson(theta L / 2)
with L the total tree length (coalescent in rescaled time Lambda, real time t = log(1 + g Lambda) / g, as in
ref_posterior_qmc.py and const_growth.py). Only S is used, so only tree lengths are simulated.
Encoder: log(1 + S), standardised. One marginal quantile network per parameter, train() of const_growth.py
(the network of hap_train.py: 2 x 128 ReLU, Adam, check loss, level input logit(u) / 2, decaying learning
rate, 60 passes); theta on the log scale, g on logit(g / 20). Sizes: 1.2e5 training replicates (as for
Table coal:tab:growth), 2e4 calibration and 2e4 test replicates (Section coal:sec:repair).
Data seeds are fixed (train 3301, calibration 3302, test 3303).

Scoring on the test set: mean 90% width, coverage at 90 and 50 percent, SBC chi square on nine degrees of
freedom (rank of the truth among 99 fitted quantiles, ten bins, expected count 2000), and coverage at 90
percent within thirds of S. Repair: V = F(theta | S) from the quantile curve on a 400 point level grid,
G the empirical distribution function of V on the calibration set, corrected quantiles h(S, G^{-1}(tau)),
scored the same way on the test set. GBC quantiles at S = 40, 100, 200 and levels 5, 25, 50, 75, 95 are
compared with the reference posterior in ref_posterior_qmc.json (written by ref_posterior_qmc.py).

Usage: python3 -I code/coal_rev/ref_gbc.py SEED
Writes ref_gbc_seed<SEED>.json (results, unrounded) and ref_gbc_weights_seed<SEED>.npz (weights and input
standardisation) beside this script, and ref_gbc_data.npz (training, calibration and test sets) once.
The chapter uses training seed 0, fixed before any result was seen; seeds 1 to 3 measure run to run spread.
"""
import sys, json, time, numpy as np
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from const_growth import train, summarise, LEVELS, SHAPE, SCALE, GMAX, n

ks = np.arange(n, 1, -1).astype(float)
rates = ks * (ks - 1) / 2
P = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
S_LIST = [40, 100, 200]
UGRID = (np.arange(400) + 0.5) / 400


def simulate(M, rng):
    th = rng.gamma(SHAPE, SCALE, M)
    g = rng.uniform(0, GMAX, M)
    Lam = np.cumsum(rng.standard_exponential((M, n - 1)) / rates, axis=1)
    safe = np.where(g > 1e-10, g, 1.0)[:, None]
    tau = np.where(g[:, None] > 1e-10, np.log1p(g[:, None] * Lam) / safe, Lam)
    L = np.diff(tau, axis=1, prepend=0.0) @ ks
    S = rng.poisson(th * L / 2)
    return th, g, S


def feat(S):
    return np.log1p(np.asarray(S, float))[:, None]


def thirds_cov(Q, truth, S, cuts):
    lo, hi = Q[:, 4], Q[:, 94]
    inside = (truth > lo) & (truth < hi)
    grp = np.digitize(S, cuts)
    return [float(inside[grp == k].mean()) for k in range(3)]


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    th, g, S = simulate(120000, np.random.default_rng(3301))
    cth, cg, cS = simulate(20000, np.random.default_rng(3302))
    tth, tg, tS = simulate(20000, np.random.default_rng(3303))
    print("mean S train %.2f cal %.2f test %.2f" % (S.mean(), cS.mean(), tS.mean()), flush=True)
    if not (_HERE / "ref_gbc_data.npz").exists():
        np.savez_compressed(str(_HERE / "ref_gbc_data.npz"), train_th=th, train_g=g, train_S=S,
                            cal_th=cth, cal_g=cg, cal_S=cS, test_th=tth, test_g=tg, test_S=tS)
    cuts = np.quantile(tS, [1 / 3, 2 / 3])
    ref = None
    if (_HERE / "ref_posterior_qmc.json").exists():
        ref = json.load(open(str(_HERE / "ref_posterior_qmc.json")))
    res = dict(seed=seed, meanS_test=float(tS.mean()), S_thirds_cuts=cuts.tolist()); weights = {}
    gg = np.clip(g / GMAX, 1e-6, 1 - 1e-6)
    specs = [("theta", np.log(th), cth, tth, np.exp), ("g", np.log(gg / (1 - gg)), cg, tg, lambda z: GMAX / (1 + np.exp(-z)))]
    for pname, y, ctruth, ttruth, inv in specs:
        t0 = time.time()
        pred, wts = train(feat(S), y, seed)
        print("%s trained in %.1f s" % (pname, time.time() - t0), flush=True)
        for a_, v_ in wts.items():
            weights["%s_%s" % (pname, a_)] = v_
        r = {}
        Q = inv(pred(feat(tS), LEVELS))
        r["before"] = summarise("test %s before" % pname, Q, ttruth)
        r["before"]["cov90_thirds"] = thirds_cov(Q, ttruth, tS, cuts)
        # repair: V on the 400 point grid, G from the calibration set
        Cc = inv(pred(feat(cS), UGRID)); Ct = inv(pred(feat(tS), UGRID))
        Vc = np.array([np.interp(ctruth[i], Cc[i], UGRID) for i in range(len(ctruth))])
        Vt = np.array([np.interp(ttruth[i], Ct[i], UGRID) for i in range(len(ttruth))])
        Ginv = np.quantile(Vc, LEVELS)
        Qr = np.array([np.interp(Ginv, UGRID, Ct[i]) for i in range(len(ttruth))])
        r["after"] = summarise("test %s recalibrated" % pname, Qr, ttruth)
        r["after"]["cov90_thirds"] = thirds_cov(Qr, ttruth, tS, cuts)
        r["Ginv_at_05_95"] = [float(Ginv[4]), float(Ginv[94])]
        # GBC quantiles at fixed S against the reference posterior
        Qs = inv(pred(feat(S_LIST), P))
        r["gbc"] = {str(s): Qs[i].tolist() for i, s in enumerate(S_LIST)}
        if ref is not None:
            R = np.array([ref[str(s)][pname] for s in S_LIST]); D = Qs - R
            k = np.unravel_index(np.abs(D).argmax(), D.shape)
            kr = np.unravel_index(np.abs(D / R).argmax(), D.shape)
            r["max_abs"] = dict(value=float(abs(D[k])), S=S_LIST[k[0]], level=float(P[k[1]]), rel=float(abs(D[k] / R[k])))
            r["max_rel"] = dict(value=float(abs(D[kr] / R[kr])), S=S_LIST[kr[0]], level=float(P[kr[1]]), absdiff=float(abs(D[kr])))
            print(pname, "max abs", r["max_abs"], "max rel", r["max_rel"])
        print(pname, "cov90 thirds before", r["before"]["cov90_thirds"], "after", r["after"]["cov90_thirds"])
        print(pname, "GBC at S:", {s: np.round(Qs[i], 2).tolist() for i, s in enumerate(S_LIST)}, flush=True)
        res[pname] = r
    np.savez_compressed(str(_HERE / ("ref_gbc_weights_seed%d.npz" % seed)), **weights)
    json.dump(res, open(str(_HERE / ("ref_gbc_seed%d.json" % seed)), "w"), indent=1)
