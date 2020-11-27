import pytest
import numpy as np

from irmetrics.topk import rr, recall, precision, ndcg


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


@pytest.mark.parametrize("y_true, y_pred, output", [
    (1, [1, 0, 0], True),
    (1, [0, 1, 0], True),
    (1, [0, 0, 1], True),
    (1, [1, 1, 1], True),
    (1, [0, 0, 0], False),
    (1, [0, 0, 0] * 20 + [1], False),
])
def test_recall(y_true, y_pred, output, n_samples=128):
    assert recall(y_true, y_pred) == output

    # Now the vectorized output
    y_trues = np.repeat(np.array(y_true), n_samples)
    y_preds = np.repeat(np.atleast_2d(y_pred), n_samples, axis=0)
    outputs = np.repeat(np.array(output), n_samples)
    assert np.all(recall(y_trues, y_preds) == outputs)


@pytest.mark.parametrize("y_true, y_pred, output", [
    (1, [1, 0, 0], 1. / 3.),
    (1, [0, 1, 0], 1. / 3.),
    (1, [0, 0, 1], 1. / 3.),
    (1, [1, 1, 1], 1. / 3.),
    (1, [0, 0, 0], False),
    (1, [0, 0, 0] * 20 + [1], False),
])
def test_precision(y_true, y_pred, output, n_samples=128):
    assert precision(y_true, y_pred) == output

    # Now the vectorized output
    y_trues = np.repeat(np.array(y_true), n_samples)
    y_preds = np.repeat(np.atleast_2d(y_pred), n_samples, axis=0)
    outputs = np.repeat(np.array(output), n_samples)
    assert np.all(precision(y_trues, y_preds) == outputs)


@pytest.mark.parametrize("y_true, y_pred, output", [
    (1, [1, 0, 0], 1.),
    (1, [0, 1, 0], 1. / np.log2(2 + 1)),  # idx + 1, where idx starts from 1
    (1, [0, 0, 1], 1. / np.log2(3 + 1)),  # idx + 1, where idx starts from 1
    (1, [1, 1, 1], 1.),
    (1, [0, 0, 0], np.nan),
    (1, [0, 0, 0] * 20 + [1], np.nan),
])
def test_ndcg(y_true, y_pred, output, n_samples=128):
    np.testing.assert_allclose(ndcg(y_true, y_pred), output)

    # Now the vectorized output
    y_trues = np.repeat(np.array(y_true), n_samples)
    y_preds = np.repeat(np.atleast_2d(y_pred), n_samples, axis=0)
    outputs = np.repeat(np.array(output), n_samples)
    np.testing.assert_allclose(ndcg(y_trues, y_preds), outputs)
