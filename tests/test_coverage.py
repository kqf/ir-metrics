import pytest
import numpy as np

from contextlib import contextmanager
from irmetrics.coverage import coverage, iou


@contextmanager
def does_not_raise():
    yield


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
def test_coverage(y_pred, output, n_samples=128):
    assert coverage(y_pred) == output

    # Now the vectorized output
    y_preds = np.repeat(np.atleast_2d(y_pred), n_samples, axis=0)
    outputs = np.repeat(np.array(output), n_samples)
    assert np.all(coverage(y_preds) == outputs)


@pytest.mark.parametrize(
    "y_true, y_pred, output, exception",
    [
        (1, [1, 2, 3], 1. / 3, does_not_raise()),
        (1, [2, 1, 3], 1. / 3, does_not_raise()),
        (1, [2, 3, 1], 1. / 3, does_not_raise()),
        (1, [3, 4, 5], 0, does_not_raise()),
        (1, [1, 1, 1], 0, pytest.raises(ValueError)),
        ([1, 1, 1], 1, 0, pytest.raises(ValueError)),
    ])
def test_iou(y_true, y_pred, output, exception, n_samples=128):
    with exception:
        assert iou(y_true, y_pred) == output


@pytest.mark.parametrize("y_true, y_pred, output, exception", [
    (1, [1, 2, 3], 1. / 3, does_not_raise()),
    (1, [2, 1, 3], 1. / 3, does_not_raise()),
    (1, [2, 3, 1], 1. / 3, does_not_raise()),
    (1, [3, 4, 5], 0, does_not_raise()),
    (1, [1, 1, 1], 0, pytest.raises(ValueError)),
    ([1, 1, 1], 1, 0, pytest.raises(ValueError)),
])
def test_iou_vectorized(y_true, y_pred, output, exception, n_samples=128):
    y_trues = np.repeat(np.array(y_true), n_samples)
    y_preds = np.repeat(np.atleast_2d(y_pred), n_samples, axis=0)
    outputs = np.repeat(np.array(output), n_samples)
    with exception:
        np.testing.assert_array_equal(iou(y_trues, y_preds), outputs)
