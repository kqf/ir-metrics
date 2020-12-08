import pytest
import numpy as np

from irmetrics.topk import rr, recall, precision, ndcg, ap
from irmetrics.relevance import unilabel, multilabel


@pytest.fixture
def cases(inputs, expected, exceptions):
    return zip(inputs, expected, exceptions)


@pytest.mark.parametrize("measure", [
    rr,
    recall,
    precision,
    ndcg,
    ap,
])
@pytest.mark.parametrize("relevance", [
    unilabel,
    multilabel,
])
def test_all(cases, measure, relevance):
    for (y_true, y_pred), expected, exception in cases:
        with exception():
            np.testing.assert_equal(
                measure(y_true, y_pred, relevance=relevance),
                expected
            )


@pytest.mark.parametrize("measure", [
    rr,
    recall,
    precision,
    ndcg,
    ap,
])
@pytest.mark.parametrize("relevance", [
    unilabel,
    multilabel,
])
def test_all_vectorized(cases, measure, relevance, n_samples=128):
    for (y_true, y_pred), expected, exception in cases:

        # Repeat the inputs and expected values along the batch dimension
        y_trues = np.repeat(np.array(y_true), n_samples)
        y_preds = np.repeat(np.atleast_2d(y_pred), n_samples, axis=0)
        outputs = np.repeat(np.array(expected), n_samples)

        with exception():
            np.testing.assert_equal(
                measure(y_trues, y_preds, relevance=relevance),
                outputs
            )
