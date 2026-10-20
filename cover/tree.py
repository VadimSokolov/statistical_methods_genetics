"""Kingman coalescent tree and GEM(3) stick breaking strip for the cover, written as TikZ code.

Usage: python3 -I cover/tree.py [seed] [--out DIR] [--force]

Writes tree.tex and strip.tex into DIR (default: the current directory) and prints where it wrote them. It refuses to overwrite an existing file: it names the file that blocks it and exits with status 1, unless --force is given. The seed is a non negative integer (default 1971).

Seed 1971, the default, reproduces the shipped cover/tree.tex and cover/strip.tex byte for byte with NumPy 2.4.2. It is the only seed from 0 to 20000 that does (exhaustive search, 2026-10-17), and the earlier default 1972 does not. NumPy does not promise identical Generator streams across versions, so with another version any seed may give different art. The shipped tree.tex and strip.tex remain the source of truth for the cover.
"""
import argparse
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("tree.py: numpy is required")

DEFAULT_SEED = 1971
N = 18             # leaves of the tree
W, HT = 5.6, 3.4   # inches of the drawing
THETA = 3.0        # GEM parameter of the strip
USAGE = "python3 -I cover/tree.py [seed] [--out DIR] [--force]"


def make_tree(rng):
    """Draw a Kingman coalescent with N leaves. Return the tikz text, the root position over N-1 and the root height."""
    n = N
    t = 0.0
    xpos = {i: i for i in range(n)}
    ht = {i: 0.0 for i in range(n)}
    nodes = n
    edges = []
    children = {}
    active = list(range(n))
    while len(active) > 1:
        k = len(active)
        t += rng.exponential(1.0 / (k * (k - 1) / 2))
        i, j = sorted(rng.choice(len(active), 2, replace=False))
        a, b = active[i], active[j]
        new = nodes
        nodes += 1
        ht[new] = t
        edges.append((a, new))
        edges.append((b, new))
        xpos[new] = None
        active = [x for x in active if x not in (a, b)] + [new]
        children[new] = (a, b)
    root = active[0]
    # leaf order by DFS so the tree is planar
    order = []

    def dfs(v):
        if v < n:
            order.append(v)
            return
        a, b = children[v]
        dfs(a)
        dfs(b)
    dfs(root)
    for k, v in enumerate(order):
        xpos[v] = k

    def X(v):
        if xpos[v] is None:
            a, b = children[v]
            xpos[v] = 0.5 * (X(a) + X(b))
        return xpos[v]
    X(root)
    H = ht[root]
    sx = W / (n - 1)
    sy = HT / H
    out = []
    for v in range(nodes):
        if v == root:
            continue
        p = [p for (c, p) in edges if c == v][0]
        x0, y0 = X(v) * sx, ht[v] * sy
        x1, y1 = X(p) * sx, ht[p] * sy
        out.append(r"\draw[treeline] (%.3f,%.3f) -- (%.3f,%.3f) -- (%.3f,%.3f);" % (x0, -y0, x0, -y1, x1, -y1))
    out.append(r"\draw[treeline] (%.3f,%.3f) -- (%.3f,%.3f);" % (X(root) * sx, -H * sy, X(root) * sx, -H * sy - 0.6))
    for v in range(n):
        out.append(r"\fill[gold] (%.3f,0) circle (0.035);" % (X(v) * sx))
    return "\n".join(out), X(root) / (n - 1), H


def make_strip(rng):
    """Draw 40 GEM(THETA) stick breaking weights plus the remainder. Return the tikz text."""
    rem = 1.0
    w = []
    for _ in range(40):
        v = rng.beta(1, THETA)
        w.append(rem * v)
        rem *= 1 - v
    w.append(rem)
    x = 0.0
    out = []
    cols = ["gold", "rust", "sage", "cream", "teal"]
    for k, wi in enumerate(w):
        out.append(r"\fill[%s] (%.4f,0) rectangle (%.4f,0.16);" % (cols[k % 5], x * W, (x + wi) * W))
        x += wi
    return "\n".join(out)


def generate(seed):
    """Return (tree.tex text, strip.tex text, root position over N-1, root height) for a seed."""
    rng = np.random.default_rng(seed)
    tree, bal, height = make_tree(rng)
    strip = make_strip(rng)
    return tree, strip, bal, height


def main(argv):
    parser = argparse.ArgumentParser(prog="tree.py", usage=USAGE, description="Write tree.tex and strip.tex for the cover.")
    parser.add_argument("seed", nargs="?", type=int, default=DEFAULT_SEED, help="non negative integer seed (default %d)" % DEFAULT_SEED)
    parser.add_argument("--out", metavar="DIR", default=".", help="output directory (default: the current directory)")
    parser.add_argument("--force", action="store_true", help="overwrite existing tree.tex and strip.tex")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be a non negative integer")
    targets = [os.path.join(args.out, "tree.tex"), os.path.join(args.out, "strip.tex")]
    blocked = [p for p in targets if os.path.lexists(p)]
    if blocked and not args.force:
        for p in blocked:
            print("tree.py: %s exists, not overwriting it (use --force to overwrite)" % os.path.abspath(p), file=sys.stderr)
        return 1
    tree, strip, bal, height = generate(args.seed)
    try:
        os.makedirs(args.out, exist_ok=True)
        for path, text in zip(targets, (tree, strip)):
            # exclusive creation unless forced, so an existing file is never truncated by accident
            with open(path, "w" if args.force else "x", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            print("wrote", os.path.abspath(path))
    except OSError as e:
        print("tree.py: cannot write: %s" % e, file=sys.stderr)
        return 1
    print("seed %d, numpy %s" % (args.seed, np.__version__))
    print("BAL", bal, height)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
