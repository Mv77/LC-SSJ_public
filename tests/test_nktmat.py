import numpy as np

from Tools.SimpleNKAgent import NKtmat

# Precomputed example matrices satisfying the transition matrix constraint.
A_TMAT = np.array(
    [
        [[0.6, 0.4], [0.2, 0.8], [0.5, 0.5]],
        [[0.3, 0.7], [0.4, 0.6], [0.9, 0.1]],
    ]
)
Z_TMAT = np.array(
    [
        [0.7, 0.2, 0.1],
        [0.1, 0.3, 0.6],
    ]
)
D_GRID = np.array(
    [
        [0.2, 0.3, 0.5],
        [0.5, 0.3, 0.2],
    ]
)
OUT_VALUES = np.array(
    [
        [1.0, 0.0],
        [0.5, 0.5],
        [0.2, 0.8],
    ]
)

ADVANCE_EXPECTED = np.array(
    [
        [0.346, 0.454],
        [0.221, 0.279],
        [0.313, 0.387],
    ]
)
EXPECT_EXPECTED = np.array(
    [
        [0.564, 0.308, 0.5],
        [0.552, 0.526, 0.396],
    ]
)


def test_transition_axes_are_stochastic():
    """Ensure the constructed transition matrices are valid."""
    assert np.allclose(A_TMAT.sum(axis=-1), 1.0)
    assert np.allclose(Z_TMAT.sum(axis=-1), 1.0)


def test_advance_matches_precomputed_values():
    tmat = NKtmat(A_TMAT, Z_TMAT)
    result = tmat.advance(D_GRID)
    np.testing.assert_allclose(result, ADVANCE_EXPECTED, atol=1e-12)


def test_expect_matches_precomputed_values():
    tmat = NKtmat(A_TMAT, Z_TMAT)
    result = tmat.expect(OUT_VALUES)
    np.testing.assert_allclose(result, EXPECT_EXPECTED, atol=1e-12)
