"""Write the 1000andUniform drop: Tom1000 plus the uniform matrix, target 2400.

    pixi run -e ml python -m tm_ml.uniform.make_drop

U is the well-mixed chain, 1/19 off the diagonal and 0 on it. Its target, 2400,
sits above the slides' a_max = 2000 and the largest measured target, 2041.6:
the label says U is the top of the range, and by a margin.

The split is a seeded permutation of n, so adding a matrix reshuffles every
split, and U lands wherever its index falls. It is inserted at the first
position that is in the training split for seeds 0, 1 and 2 at the default
fractions, so every one of those seeds trains on it. The 1000 Tom1000 matrices
keep their order around it. The file mirrors Tom1000's layout, an object array
of matrices and one of targets, so `ingest` reads it unchanged.
"""

import numpy as np

from tm_ml.datasets import split_indices
from tm_ml.ingest import MATRIX_KEY, RAW_DIR, TARGET_KEY

SOURCE = RAW_DIR / "Tom1000" / "Analysis_Results.npz"
DROP = "1000andUniform"
TARGET_U = 2400.0
SEEDS = (0, 1, 2)
VAL_FRAC = TEST_FRAC = 0.1


def position(n):
    train = [set(split_indices(n, VAL_FRAC, TEST_FRAC, s)["train"].tolist()) for s in SEEDS]
    return min(set.intersection(*train))


def main():
    with np.load(SOURCE, allow_pickle=True) as data:
        matrices = np.asarray(data[MATRIX_KEY].tolist(), dtype=np.float64)
        targets = np.asarray(data[TARGET_KEY].tolist(), dtype=np.float64)

    n = matrices.shape[-1]
    U = np.full((n, n), 1.0 / (n - 1))
    np.fill_diagonal(U, 0.0)
    k = position(len(matrices) + 1)
    matrices = np.insert(matrices, k, U, axis=0)
    targets = np.insert(targets, k, TARGET_U)

    out = RAW_DIR / DROP
    out.mkdir(exist_ok=True)
    np.savez(out / "Analysis_Results.npz",
             **{MATRIX_KEY: matrices.astype(object), TARGET_KEY: targets.astype(object)})
    print(f"wrote {out}: {len(matrices)} matrices, U at index {k} with target {TARGET_U:g}")


if __name__ == "__main__":
    main()
