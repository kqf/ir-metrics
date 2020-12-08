import pytest
import numpy as np
import pandas as pd
from irmetrics.flat import flat
from irmetrics.topk import rr, ndcg


@pytest.fixture
def data(inputs):
    # Explode the examples
    df = pd.DataFrame([
        {
            "query": i,
            "relevance": answer == prediction,
            "weights": 1,
        }
        for i, (answer, predictions) in enumerate(inputs)
        for prediction in predictions
    ])
    return df


@pytest.mark.parametrize("measure", [
    rr,
    ndcg,
    # Not going to support these methods as they require
    # true shapes of y_pred/y_true.
    # recall,
    # precision,
    # ap,
])
def test_calculates_data(data, expected, measure):
    outputs = flat(
        data,
        query_col="query",
        relevance_col="relevance",
        measure=measure,
    )
    np.testing.assert_almost_equal(outputs.values, expected)


"""
    This is a separate test for NDCG as it might accept nonbinary judgements
"""

EXAMPLES = [
    ((3, 2, 3, 0, 1, 2), 0.948810748),
    ((1, 0, 0, 0, 0, 0), 1.0),
    ((0, 1, 0, 0, 0, 0), 0.630929753),
    ((1, 1, 0, 0, 0, 0), 1.0),
    ((0, 0, 0, 0, 0, 1), 0.356207187),
    ((0, 0, 0, 0, 0, 0), np.NaN)
]


@pytest.fixture
def nonbinary_data():
    queries, expected = zip(*EXAMPLES)
    # Explode the examples
    df = pd.DataFrame([
        {
            "query": i,
            "relevance": rel,
            "weights": 1,
        }
        for i, relevance in enumerate(queries)
        for rel in relevance
    ])
    return df, expected


def test_nonbinary_ndcg(nonbinary_data):
    df, expected = nonbinary_data
    outputs = flat(
        df,
        query_col="query",
        relevance_col="relevance",
        measure=ndcg,
    )
    np.testing.assert_almost_equal(outputs.values, expected)
