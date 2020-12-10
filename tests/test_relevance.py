import pytest
import numpy as np

from irmetrics.relevance import unilabel, multilabel


def arr(x):
    return np.array([x])


@pytest.mark.parametrize("y_true, y_pred, output", [
    (arr([1]), arr([1, 2, 3]), arr([True, False, False])),
    (arr([1]), arr([2, 2, 3]), arr([False, False, False])),
])
@pytest.mark.parametrize("relevance", [unilabel, multilabel])
def test_single_labels(y_true, y_pred, output, relevance):
    np.testing.assert_equal(relevance(y_true, y_pred), output)
