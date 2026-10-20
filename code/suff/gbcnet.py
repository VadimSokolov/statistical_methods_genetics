"""Shared pieces for the sufficiency chapter figures.

Deep sets quantile network H(u, y) = g(u, sum_i phi(y_i)) fitted by the check loss
(Generative Bayesian Computation), implemented in JAX, plus the house plotting style.
"""
import os as _os, sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "book.tex").exists())
_FIGDIR = _Path(_os.environ.get("SMG_FIG_DIR", _ROOT / "figures"))
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import jax
import jax.numpy as jnp
import optax

FIGDIR = str(_FIGDIR)


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })
    return plt


# muted colourblind safe palette (Tol muted)
COL = {"blue": "#332288", "teal": "#44AA99", "rose": "#CC6677", "sand": "#DDCC77",
       "cyan": "#88CCEE", "wine": "#882255", "green": "#117733", "grey": "#777777",
       "olive": "#999933"}


def _mlp_init(key, sizes):
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a)
        params.append((W, jnp.zeros(b)))
    return params


def _mlp(params, x):
    for W, b in params[:-1]:
        x = jax.nn.silu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def init_deepsets(key, d, feat_dim=2, hidden=64, plain_input=False):
    """phi: feat_dim -> d (three layers), g: 1 + d -> 1 (three layers).
    If plain_input, there is no phi; g sees (u, x) with x a fixed summary of size d."""
    k1, k2 = jax.random.split(key)
    p = {"g": _mlp_init(k2, [1 + d, hidden, hidden, 1])}
    if not plain_input:
        p["phi"] = _mlp_init(k1, [feat_dim, hidden, hidden, d])
    return p


def pooled(params, feats, mask):
    """feats (B, L, f), mask (B, L) -> T (B, d), masked sum over set elements."""
    z = _mlp(params["phi"], feats)
    return jnp.sum(z * mask[..., None], axis=1)


def H(params, u, feats, mask, plain_input=False):
    T = feats if plain_input else pooled(params, feats, mask)
    x = jnp.concatenate([u[:, None], T], axis=1)
    return _mlp(params["g"], x)[:, 0]


def check_loss(z, u):
    return z * (u - (z < 0))


def fit(feats, mask, target, d, steps, key, plain_input=False, batch=256, lr=3e-3,
        hidden=64, verbose=False):
    """Fit H by stochastic minimisation of the check loss with fresh u each step.
    Inputs are numpy arrays; returns params."""
    feats = jnp.asarray(feats, jnp.float32)
    mask = None if mask is None else jnp.asarray(mask, jnp.float32)
    target = jnp.asarray(target, jnp.float32)
    key, k0 = jax.random.split(key)
    params = init_deepsets(k0, d, feat_dim=feats.shape[-1], hidden=hidden,
                           plain_input=plain_input)
    sched = optax.cosine_decay_schedule(lr, steps)
    opt = optax.adam(sched)
    state = opt.init(params)
    N = target.shape[0]

    def loss_fn(p, f, m, t, u):
        return jnp.mean(check_loss(t - H(p, u, f, m, plain_input), u))

    @jax.jit
    def step(p, s, k):
        ki, ku = jax.random.split(k)
        idx = jax.random.randint(ki, (batch,), 0, N)
        f = feats[idx]
        m = None if mask is None else mask[idx]
        u = jax.random.uniform(ku, (batch,))
        l, g = jax.value_and_grad(loss_fn)(p, f, m, target[idx], u)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, l

    for it in range(steps):
        key, k = jax.random.split(key)
        params, state, l = step(params, state, k)
        if verbose and it % 200 == 0:
            print(it, float(l), flush=True)
    return params


def heldout_risk(params, feats, mask, target, key, plain_input=False, ndraws=6):
    feats = jnp.asarray(feats, jnp.float32)
    mask = None if mask is None else jnp.asarray(mask, jnp.float32)
    target = jnp.asarray(target, jnp.float32)
    tot = 0.0
    for _ in range(ndraws):
        key, k = jax.random.split(key)
        u = jax.random.uniform(k, (target.shape[0],))
        tot += float(jnp.mean(check_loss(target - H(params, u, feats, mask, plain_input), u)))
    return tot / ndraws


def quantiles(params, feats, mask, levels, plain_input=False):
    feats = jnp.asarray(feats, jnp.float32)
    mask = None if mask is None else jnp.asarray(mask, jnp.float32)
    out = []
    for q in levels:
        u = jnp.full((feats.shape[0],), q, jnp.float32)
        out.append(np.asarray(H(params, u, feats, mask, plain_input)))
    return np.stack(out, axis=1)


def bottleneck(params, feats, mask):
    return np.asarray(pooled(params, jnp.asarray(feats, jnp.float32),
                             jnp.asarray(mask, jnp.float32)))


def grid_quantiles(grid, w, levels):
    """Quantiles of a discrete law on grid with weights w (rows)."""
    c = np.cumsum(w, axis=-1)
    c = c / c[..., -1:]
    return np.stack([np.array([np.interp(q, ci, grid) for ci in np.atleast_2d(c)])
                     for q in levels], axis=-1)
