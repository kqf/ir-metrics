import pytest
import numpy as np

from contextlib import contextmanager
from irmetrics.topk import rr, recall, precision, ndcg, ap


@contextmanager
def does_not_raise():
    yield


@pytest.fixture
def inputs():
    return [
        (1, [1, 2, 3]),
        (1, [2, 1, 3]),
        (1, [2, 3, 1]),
        (1, [1, 2, 3]),
        (1, [np.nan, np.nan, np.nan]),
        (1, [8, 9, 10]),  # all irrelevant
        ("item1", ["item1", "item2", "item3"]),
        ("item1", ["item1", np.nan, np.nan]),  # noqa, raises as nans are converted to str
        # (1, [1, None, None]),
    ]


_OUTPUTS = {
    rr: [
        (1, does_not_raise),
        (1. / 2, does_not_raise),
        (1. / 3, does_not_raise),
        (1., does_not_raise),
        (0, does_not_raise),
        (0, does_not_raise),
        (1, does_not_raise),
        (1, does_not_raise),
        # (1, does_not_raise),
    ],
    recall: [
        (1, does_not_raise),
        (1, does_not_raise),
        (1, does_not_raise),
        (1, does_not_raise),
        (0, does_not_raise),
        (0, does_not_raise),
        (1, does_not_raise),
        (1, does_not_raise),
        # (1, does_not_raise),
    ],
    precision: [
        (1. / 3., does_not_raise),
        (1. / 3., does_not_raise),
        (1. / 3., does_not_raise),
        (1. / 3., does_not_raise),
        (0., does_not_raise),
        (0., does_not_raise),
        (1. / 3., does_not_raise),
        (1. / 3., does_not_raise),
        # (1. / 3., does_not_raise),
    ],
    ndcg: [
        (1., does_not_raise),
        (1. / np.log2(2 + 1), does_not_raise),  # idx + 1, (idx starts from 1)
        (1. / np.log2(3 + 1), does_not_raise),  # idx + 1, (idx starts from 1)
        (1., does_not_raise),
        (np.nan, does_not_raise),
        (np.nan, does_not_raise),
        (1., does_not_raise),
        (1., does_not_raise),
        # (1., does_not_raise),
    ],
    ap: [
        (1. / 3., does_not_raise),
        (0., does_not_raise),
        (0., does_not_raise),
        (1. / 3., does_not_raise),
        (0., does_not_raise),
        (0., does_not_raise),
        (1. / 3., does_not_raise),
        (1. / 3., does_not_raise),
        # (1. / 3., does_not_raise),
    ],
}


@pytest.fixture
def raw_outputs(measure):
    return list(zip(*_OUTPUTS[measure]))


@pytest.fixture
def expected(raw_outputs):
    return raw_outputs[0]


@pytest.fixture
def exceptions(raw_outputs):
    return raw_outputs[1]
