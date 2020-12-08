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
def test_calculates_data(data, outputs, expected, measure):
    out = flat(
        data,
        query_col="query",
        relevance_col="relevance",
        measure=measure,
    )
    np.testing.assert_almost_equal(out.values, outputs)
