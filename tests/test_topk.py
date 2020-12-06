import pytest
import numpy as np

from contextlib import contextmanager

from irmetrics.topk import rr, recall, precision, ndcg, ap
from irmetrics.relevance import unilabel, multilabel


@contextmanager
def does_not_raise():
    yield


INPUTS = [
    (1, [1, 0, 0]),
    (1, [0, 1, 0]),
    (1, [0, 0, 1]),
    (1, [1, 1, 1]),
    (1, [0, 0, 0]),
    (1, [0, 0, 0]),
]

OUTPUTS = {
    rr: [
        (1, does_not_raise),
        (1. / 2, does_not_raise),
        (1. / 3, does_not_raise),
        (1., does_not_raise),
        (0, does_not_raise),
        (0, does_not_raise),
    ],
    recall: [
        (True, does_not_raise),
        (True, does_not_raise),
        (True, does_not_raise),
        (True, does_not_raise),
        (False, does_not_raise),
        (False, does_not_raise),
    ],
    precision: [
        (1. / 3., does_not_raise),
        (1. / 3., does_not_raise),
        (1. / 3., does_not_raise),
        (1. / 3., does_not_raise),
        (False, does_not_raise),
        (False, does_not_raise),
    ],
    ndcg: [
        (1., does_not_raise),
        (1. / np.log2(2 + 1), does_not_raise),  # idx + 1, (idx starts from 1)
        (1. / np.log2(3 + 1), does_not_raise),  # idx + 1, (idx starts from 1)
        (1., does_not_raise),
        (np.nan, does_not_raise),
        (np.nan, does_not_raise),
    ],
    ap: [
        (1. / 3., does_not_raise),
        (0., does_not_raise),
        (0., does_not_raise),
        (1. / 3., does_not_raise),
        (False, does_not_raise),
        (False, does_not_raise),
    ],
}


def _pars(keys, inputs, outputs):
    params = []
    for key in keys:
        for arguments, exp in zip(inputs, outputs[key]):
            params.append(arguments + exp + (key,))
    return params


@pytest.mark.parametrize(
    "y_true, y_pred, output, expectation, f",
    _pars([
        rr,
        recall,
        precision,
        ndcg,
        ap,
    ], INPUTS, OUTPUTS)
)
@pytest.mark.parametrize("relevance", [
    unilabel,
    multilabel,
])
def test_all(y_true, y_pred, output, expectation, f, relevance, n_samples=128):
    with expectation():
        np.testing.assert_equal(f(y_true, y_pred, relevance=relevance), output)

    # Now the vectorized version of the same function
    y_trues = np.repeat(np.array(y_true), n_samples)
    y_preds = np.repeat(np.atleast_2d(y_pred), n_samples, axis=0)
    outputs = np.repeat(np.array(output), n_samples)

    with expectation():
        np.testing.assert_equal(
            f(y_trues, y_preds, relevance=relevance), outputs)
