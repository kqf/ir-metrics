import pytest
import numpy as np
import pandas as pd
from irmetrics.flat import flat
from irmetrics.topk import ndcg

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
def data():
    queries, answers = zip(*EXAMPLES)
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
    return df, answers


def test_calculates_data(data):
    df, answers = data
    outputs = flat(
        df,
        query_col="query",
        relevance_col="relevance",
        measure=ndcg,
    )
    np.testing.assert_almost_equal(outputs.values, answers)
