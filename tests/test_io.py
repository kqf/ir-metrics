import pytest
import numpy as np

from numpy import array as ar
from irmetrics.io import ensure_inputs


# Identity shortcut
_id = ar([[1]])


@pytest.mark.parametrize("y_true, y_pred, y_true_ex, y_pred_ex", [
    (1, 1, _id, _id),  # sc -> [1, 1]
    # First check only y_pred
    (1, [1], _id, _id),  # [1] -> [1, 1]
    (1, [[1]], _id, _id),  # [1, 1] -> [1, 1]
    (1, [[1, 2]], _id, ar([[1, 2]])),  # [1, 2] -> [1, 2]
    (
        1, np.tile(1, (128, 40)),
        _id, np.tile(1, (128, 20))
    ),  # [n_samples, n_preds] -> [n_samples, k]
])
def test_handles_inputs(y_true, y_pred, y_true_ex, y_pred_ex):
    true, pred = ensure_inputs(y_true, y_pred)

    # First check shapes (np does broadcasting for its testing functions)
    assert true.shape == y_true_ex.shape
    assert pred.shape == y_pred_ex.shape

    np.testing.assert_array_equal(true, y_true_ex)
    np.testing.assert_array_equal(pred, y_pred_ex)
