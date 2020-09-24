import pytest
import numpy as np
from irmetrics.ranking import ndcg_score


EXAMPLES = [
    ((3, 2, 3, 0, 1, 2), 0.948810748),
    ((1, 0, 0, 0, 0, 0), 1.0),
    ((0, 1, 0, 0, 0, 0), 0.630929753),
    ((1, 1, 0, 0, 0, 0), 1.0),
    ((0, 0, 0, 0, 0, 1), 0.356207187),
    ((0, 0, 0, 0, 0, 0), np.NaN)
]


@pytest.mark.parametrize("data, answer", EXAMPLES)
def test_ndcg_score(data, answer):
    np.testing.assert_approx_equal(ndcg_score(data), answer)
