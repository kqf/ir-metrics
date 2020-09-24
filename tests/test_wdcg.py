import pytest
import numpy as np
import pandas as pd
from irmetrics.ranking import wdcg_score, mean_wdcg_score


EXAMPLES = [
    ((3, 2, 3, 0, 1, 2), 0.948810748),
    ((1, 0, 0, 0, 0, 0), 1.0),
    ((0, 1, 0, 0, 0, 0), 0.630929753),
    ((1, 1, 0, 0, 0, 0), 1.0),
    ((0, 0, 0, 0, 0, 1), 0.356207187),
    ((0, 0, 0, 0, 0, 0), np.NaN)
]


@pytest.mark.skip
@pytest.mark.parametrize("data, answer", EXAMPLES)
def test_wdcg_score(data, answer):
    np.testing.assert_approx_equal(wdcg_score(data, (1,) * len(data)), answer)


@pytest.fixture
def data():
    queries, answers = zip(*EXAMPLES)
    return pd.DataFrame([
        {
            "query": i,
            "relevance": rel,
            "weights": 1,
        }
        for i, relevance in enumerate(queries)
        for rel in relevance
    ])


def test_calculates_data(data):
    mean_wdcg_score(
        data,
        query_col="query",
        relevance_col="relevance",
        weights_col="weights")
