"""Simulate training and test sets for the haplotype experiment: n=30, theta ~ Ga(2, scale 4).
Records spectrum, S, the true haplotype count H, and the chapter's flagless recursion proxy."""
import sys, numpy as np, time

def sim_one(n, theta, rng):
    lin = [1 << i for i in range(n)]
    mut = [0]*n             # mutations on the current branch of each lineage since its birth
    node = [(1, 1)]*n        # (c, ancestral type present) correct recursion
    nodef = [1]*n            # flagless recursion
    sites = []
    for k in range(n, 1, -1):
        T = rng.exponential(2.0/(k*(k-1)))
        m = rng.poisson(theta*T/2, size=k)
        for idx in range(k):
            c = int(m[idx])
            if c:
                mut[idx] += c
                sites.extend([lin[idx]]*c)
        i, j = rng.choice(k, size=2, replace=False)
        fa, fb = mut[i] == 0, mut[j] == 0
        (cA, aA), (cB, aB) = node[i], node[j]
        newc = (cA + cB - int(aA and aB and fa and fb), int((aA and fa) or (aB and fb)))
        newf = nodef[i] + nodef[j] - int(fa and fb)
        keep = [t for t in range(k) if t not in (i, j)]
        lin = [lin[t] for t in keep] + [lin[i] | lin[j]]
        mut = [mut[t] for t in keep] + [0]
        node = [node[t] for t in keep] + [newc]
        nodef = [nodef[t] for t in keep] + [newf]
    xi = np.zeros(n-1, int)
    for b in sites:
        xi[bin(b).count("1")-1] += 1
    rows = {tuple(((b >> leaf) & 1) for b in sites) for leaf in range(n)}
    return xi, len(rows), node[0][0], nodef[0]

if __name__ == "__main__":
    n = 30; M = int(sys.argv[1]); seed = int(sys.argv[2]); out = sys.argv[3]
    rng = np.random.default_rng(seed)
    SH_, SC_ = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (2.0, 4.0)
    th = rng.gamma(SH_, SC_, M)
    XI = np.zeros((M, n-1), int); H = np.zeros(M, int); Hc = np.zeros(M, int); Hf = np.zeros(M, int)
    t0 = time.time()
    for r in range(M):
        XI[r], H[r], Hc[r], Hf[r] = sim_one(n, th[r], rng)
    print("time", time.time()-t0, "H==Hrec", np.all(H == Hc), "mean H", H.mean(), "mean flagless", Hf.mean(), "mean S", XI.sum(1).mean())
    np.savez(out, th=th, xi=XI, H=H, Hf=Hf)
