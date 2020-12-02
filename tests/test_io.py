import pytest
import numpy as np

from irmetrics.io import _ensure_io


@_ensure_io
def identity_true(y_true, y_pred, k=20):
    return y_true, y_pred


@pytest.mark.parametrize("y_true, y_pred, y_true_ex, y_pred_ex", [
    (1, 1, 1, 1),
    ([1], [1], 1, 1),
])
def test_handles_inputs(y_true, y_pred, y_true_ex, y_pred_ex):
    true, pred = identity_true(y_true, y_pred)
    np.testing.assert_equal(true, y_true_ex)
    np.testing.assert_equal(pred, y_pred_ex)
