import pytest
import numpy as np

from numpy import array as ar
from irmetrics.io import ensure_inputs


# Identity shortcut
_id = ar([[1]])
_K = 20


@pytest.mark.parametrize("y_pred, y_pred_ex", [
    (1, _id),  # scalar -> [1, 1]
    ([1], _id),  # [1] -> [1, 1]
    ([1, 2, 3, 4], ar([[1, 2, 3, 4]])),  # [n] -> [1, n]
    ([[1]], _id),  # [1, 1] -> [1, 1]
    ([[1, 2]], ar([[1, 2]])),  # [1, 2] -> [1, 2]
    ([[1], [2]], ar([[1], [2]])),  # [2, 1] -> [2, 1]
    # Realistic case: [n_samples, n_preds] -> [n_samples, k]
    (np.tile(1, (128, 40)), np.tile(1, (128, _K))),
])
@pytest.mark.parametrize("y_true, y_true_ex", [
    (1, _id),  # scalar -> [1, 1]
    ([1], _id),  # [1] -> [1, 1]
    ([[1]], _id),  # [1, 1] -> [1, 1]
    # Realistic case: [n_samples, n_preds] -> [n_samples, k]
    (np.tile(1, (128, 40)), np.tile(1, (128, _K))),
    # The cases below depend on the y_pred input
    # ([[1, 2]], ar([[1, 2]])),  # [1, 2] -> [1, 2]
    # ([[1], [2]], ar([[1], [2]])),  # [2, 1] -> [2, 1]
    # ([1, 2, 3, 4], ar([[1], [2], [3], [4]])),  # [n] -> [n, 1]
])
def test_handles_inputs(y_true, y_pred, y_true_ex, y_pred_ex):
    true, pred = ensure_inputs(y_true, y_pred, k=_K)

    # First check shapes (np does broadcasting for its testing functions)
    assert true.shape == y_true_ex.shape
    assert pred.shape == y_pred_ex.shape

    np.testing.assert_array_equal(true, y_true_ex)
    np.testing.assert_array_equal(pred, y_pred_ex)
