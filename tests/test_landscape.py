"""Node positions come back from a transition matrix the way the drop was made."""

import numpy as np

from tm_ml import landscape


def test_positions_recover_the_generating_points():
    points = np.random.default_rng(0).uniform(size=(20, 2))
    X, checks = landscape.positions(landscape.generate(points))

    assert checks["asymmetry"] < 1e-12
    assert checks["nonplanar"] < 1e-10
    assert np.abs(landscape.align(X, points) - points).max() < 1e-8


def test_to_unit_square_fills_the_square():
    points = np.random.default_rng(1).uniform(size=(20, 2))
    X = landscape.to_unit_square(points @ np.array([[0.6, -0.8], [0.8, 0.6]]) * 3.0)

    assert np.allclose(X.min(axis=0), 0.0)
    assert np.isclose(max(np.ptp(X, axis=0)), 1.0)


def test_align_undoes_a_reflection_rotation_shift_and_scale():
    reference = np.random.default_rng(2).normal(size=(20, 2))
    moved = 2.5 * reference @ np.array([[0.0, 1.0], [1.0, 0.0]]) + [3.0, -1.0]

    assert np.allclose(landscape.align(moved, reference), reference)
