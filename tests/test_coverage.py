import pytest
import numpy as np

from irmetrics.coverage import coverage, iou


@pytest.mark.parametrize("y_pred, output", [
    ([1, 0, 0], 1),
    ([0, 1, 0], 1.),
    ([0, 0, 1], 1.),
    ([1, 1, 1], 1.),
    ([0, 0, 0], 1.),
    ([0, 0, 0] * 20 + [1], 1),
    ([0, None], 1),
    ([None], 0),
])
def test_rr(y_pred, output, n_samples=128):
    assert coverage(y_pred) == output

    # Now the vectorized output
    y_preds = np.repeat(np.atleast_2d(y_pred), n_samples, axis=0)
    outputs = np.repeat(np.array(output), n_samples)
    assert np.all(coverage(y_preds) == outputs)


@pytest.mark.parametrize("y_true, y_pred, output", [
    (1, [1, 2, 3], 1. / 3),
    (1, [2, 1, 3], 1. / 3),
    (1, [2, 3, 1], 1. / 3),
    (1, [3, 4, 5], 0),
])
def test_iou(y_true, y_pred, output, n_samples=128):
    assert iou(y_true, y_pred) == output

    # Now the vectorized output
    y_trues = np.repeat(np.array(y_true), n_samples)
    y_preds = np.repeat(np.atleast_2d(y_pred), n_samples, axis=0)
    outputs = np.repeat(np.array(output), n_samples)
    assert np.all(iou(y_trues, y_preds) == outputs)
