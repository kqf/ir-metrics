import pytest
import numpy as np
from irmetrics.topk import rr


@pytest.mark.parametrize("y_true, y_pred, output", [
    (1, [1, 0, 0], 1),
    (1, [0, 1, 0], 1. / 2),
    (1, [0, 0, 1], 1. / 3),
    (1, [1, 1, 1], 1.),
    (1, [0, 0, 0], 0),
    (1, [0, 0, 0] * 20 + [1], 0),
])
def test_rr(y_true, y_pred, output, n_samples=128):
    assert rr(y_true, y_pred) == output

    # Now the vectorized output
    y_trues = np.repeat(np.array(y_true), n_samples)
    y_preds = np.repeat(np.atleast_2d(y_pred), n_samples, axis=0)
    outputs = np.repeat(np.array(output), n_samples)
    assert np.all(rr(y_trues, y_preds) == outputs)
