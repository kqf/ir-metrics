import pytest
import numpy as np

from irmetrics.relevance import unilabel, multilabel
from contextlib import contextmanager


def arr(x):
    return np.array([x])


@pytest.mark.parametrize("y_true, y_pred, output", [
    (arr([1]), arr([1, 2, 3]), arr([True, False, False])),
    (arr([1]), arr([2, 2, 3]), arr([False, False, False])),
])
@pytest.mark.parametrize("relevance", [unilabel, multilabel])
def test_single_labels(y_true, y_pred, output, relevance):
    np.testing.assert_equal(relevance(y_true, y_pred), output)


@contextmanager
def does_not_raise():
    yield


def conditional_raises(relevance):
    if relevance == unilabel:
        return pytest.raises(ValueError)
    return does_not_raise()


@pytest.mark.parametrize("y_true, y_pred, output", [
    (arr([1, 2]), arr([1, 2, 3]), arr([True, True, False])),
    (arr([1, 2]), arr([2, 2, 3]), arr([True, True, False])),
])
@pytest.mark.parametrize("relevance", [unilabel, multilabel])
def test_multiple_labels(y_true, y_pred, output, relevance):
    with conditional_raises(relevance):
        np.testing.assert_equal(relevance(y_true, y_pred), output)
