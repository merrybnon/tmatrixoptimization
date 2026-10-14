"""Node positions from a transition matrix, by inverting how the drop was made.

The drop's matrices come from ``n`` points in a unit square, symmetric weights
W_ij = D_ij⁻⁴ off the diagonal, and each row normalized: T_ij = W_ij / r_i with
r_i = Σ_k W_ik. A random walk on symmetric weights satisfies detailed balance
with stationary π_i ∝ r_i, so π_i T_ij ∝ W_ij and D_ij ∝ (π_i T_ij)^(−1/4).
Classical MDS turns those distances back into coordinates.

T fixes the layout only up to translation, rotation, reflection and one overall
scale. `to_unit_square` picks a convention for a layout on its own; `align`
places one layout onto another with the same node labels, which removes the
ambiguity entirely.

`draw` puts a layout on an axis as the source deck does, blue nodes under
directed edges whose width follows the weight.

Kept free of torch so it runs in the default environment.
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch

from tm_ml.style import INK, SURFACE, TRAIN

DISPERSAL_POWER = 4
# The widest edge, in points, drawn for the heaviest T_ij, every other edge in
# proportion and none thinner than the floor, so weak links stay visible.
EDGE_WIDTH_MAX = 4.0
EDGE_WIDTH_MIN = 0.12
# Room around the unit square, and around any layout that leaves it.
MARGIN = 0.08
# Rotations tried when squaring a layout, over the quarter turn that exhausts them.
SQUARE_ANGLES = np.deg2rad(np.arange(0.0, 90.0, 0.05))


def stationary(T):
    """The left eigenvector of T for eigenvalue 1, summing to one."""
    values, vectors = np.linalg.eig(T.T)
    pi = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    return pi / pi.sum()


def distances(T):
    """Pairwise distances implied by T, up to scale, and how far T is from
    detailed balance: max |π_i T_ij − π_j T_ji| over the largest π_i T_ij.
    Zero for every matrix in the drop; the symmetric part is used regardless."""
    n = T.shape[-1]
    off = ~np.eye(n, dtype=bool)
    W = stationary(T)[:, None] * T
    asymmetry = np.abs(W - W.T)[off].max() / W[off].max()
    W = 0.5 * (W + W.T)
    D = np.zeros_like(W)
    D[off] = np.clip(W[off], 1e-300, None) ** (-1.0 / DISPERSAL_POWER)
    return D / np.median(D[off]), asymmetry


def classical_mds(D, dims=2):
    """Coordinates whose distances best match D, and the share of the
    eigenvalue magnitude beyond ``dims``, zero when D is exactly Euclidean
    in ``dims`` dimensions."""
    n = D.shape[-1]
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    values, vectors = np.linalg.eigh(B)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    X = vectors[:, :dims] * np.sqrt(np.clip(values[:dims], 0.0, None))
    return X, np.abs(values[dims:]).sum() / np.abs(values).sum()


def to_unit_square(X):
    """Rotate so the smallest enclosing square is axis-aligned, then scale and
    shift that square onto [0, 1]². Points drawn uniformly in a square span
    about (n − 1)/(n + 1) of it, so this stretches them by roughly 10% at n = 20,
    and which of the square's eight symmetries comes out is arbitrary."""
    c, s = np.cos(SQUARE_ANGLES), np.sin(SQUARE_ANGLES)
    x = X[:, 0, None] * c - X[:, 1, None] * s
    y = X[:, 0, None] * s + X[:, 1, None] * c
    side = np.maximum(np.ptp(x, axis=0), np.ptp(y, axis=0))
    best = np.argmin(side)
    rotated = np.stack([x[:, best], y[:, best]], axis=1)
    lower = rotated.min(axis=0)
    return (rotated - lower) / side[best]


def align(X, reference, scale=True):
    """X under the translation, rotation or reflection, and optionally scale,
    that bring it closest to ``reference`` in least squares, rows matched by
    index. The fitted scale shrinks toward zero as the match worsens, so leave
    it off when X already has a scale of its own."""
    X_mean, ref_mean = X.mean(axis=0), reference.mean(axis=0)
    Xc, Rc = X - X_mean, reference - ref_mean
    U, S, Vt = np.linalg.svd(Xc.T @ Rc)
    factor = S.sum() / (Xc ** 2).sum() if scale else 1.0
    return factor * Xc @ (U @ Vt) + ref_mean


def positions(T):
    """A layout for T on the unit square, with the two checks on whether T could
    have come from the generator: its detailed-balance asymmetry and the share
    of the MDS spectrum that is not planar."""
    D, asymmetry = distances(np.asarray(T, dtype=np.float64))
    X, nonplanar = classical_mds(D)
    return to_unit_square(X), {"asymmetry": asymmetry, "nonplanar": nonplanar}


def generate(points):
    """The drop's transition matrix for the given node coordinates."""
    D = np.linalg.norm(points[:, None] - points[None], axis=-1)
    W = np.zeros_like(D)
    off = ~np.eye(len(points), dtype=bool)
    W[off] = D[off] ** -DISPERSAL_POWER
    return W / W.sum(axis=1, keepdims=True)


def limits(*layouts):
    """Shared axis limits: the unit square with a margin, grown to fit every layout."""
    every = np.concatenate(layouts)
    return (min(-MARGIN, every.min() - 0.75 * MARGIN), max(1 + MARGIN, every.max() + 0.75 * MARGIN))


def draw(ax, T, xy, heaviest, bounds, width_max=EDGE_WIDTH_MAX, node_size=150, label_size=6.5):
    """Directed edges i → j as arcs of width ∝ T_ij / ``heaviest``, the two
    directions of a pair bowing to opposite sides, under numbered nodes."""
    n = len(xy)
    shrink = np.sqrt(node_size) / 2
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            width = max(width_max * T[i, j] / heaviest, EDGE_WIDTH_MIN)
            ax.add_patch(FancyArrowPatch(
                xy[i], xy[j], connectionstyle="arc3,rad=0.2", arrowstyle="-|>",
                mutation_scale=4.0 + 2.5 * width, linewidth=width, color=INK,
                shrinkA=shrink, shrinkB=shrink, zorder=1,
            ))
    ax.scatter(xy[:, 0], xy[:, 1], s=node_size, color=TRAIN, edgecolor=SURFACE,
               linewidth=1.0, zorder=2)
    for k, (x, y) in enumerate(xy):
        ax.text(x, y, str(k), ha="center", va="center", fontsize=label_size, color=SURFACE,
                zorder=3)
    ax.set_xlim(*bounds)
    ax.set_ylim(*bounds)
    ax.set_aspect("equal")
