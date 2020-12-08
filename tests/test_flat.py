import pytest
import numpy as np
import pandas as pd
from irmetrics.flat import flat
from irmetrics.topk import rr, recall, precision, ndcg, ap


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
    recall,
    precision,
    ndcg,
    # ap,
])
def test_calculates_data(data, outputs, expected, measure):
    outputs = flat(
        data,
        query_col="query",
        relevance_col="relevance",
        measure=measure,
    )
    np.testing.assert_almost_equal(outputs.values, outputs)
