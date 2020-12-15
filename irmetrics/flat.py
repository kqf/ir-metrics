from functools import partial


def _relevance(y_true, y_pred):
    """
    This is a helper function to adjust convert the inputs for top-k metrics.

    Returns
    -------
    y_true : ndarray
        Passes through the this parameter unchanged.

    """
    return y_true


def flat(df, query_col, relevance_col, measure, k=None):
    """
    Calculate the corresponding measure for the data in flat format, with
    precalculated relevance judgements:

    +---------------+-------------------+-----------------+
    |   query_col   |   relevance_col   |   weights_col   |
    +---------------+-------------------+-----------------+
    |   1           |   0               |   1.0           |
    |   1           |   1               |   2.0           |
    |   1           |   0               |   3.0           |
    |   2           |   0               |   4.0           |
    |   2           |   1               |   5.0           |
    |   2           |   1               |   6.0           |
    |   2           |   1               |   7.0           |
    +---------------+-------------------+-----------------+

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset in the flat form: each row corresponds to a sample with the
        given query_id and relevance judgement (higher is better).
    query_col :  str
        The column that corresponds to query identificator.
    relevance_col :  str
        The column that corresponds to relevance judgements.
    measure :  callable
        The desired measure to be calculated (one from `irmetrics.topk`).
        Currently, only ``topk.ndcg`` and ``topk.rr`` are supported.
    k : int, default=None
        Only consider the highest k scores in the ranking. If None, use all
        outputs.

    Returns
    -------
    measures : pandas.core.series.Series
        The values of the corresponding measure calculated per each query.

    Examples
    --------
    >>> import pandas as pd
    >>> from irmetrics.topk import rr
    >>> from irmetrics.flat import flat
    >>> df = pd.DataFrame({"quid": [1, 1, 2, 2], "rel": [1, 0, 0, 1]})
    >>> flat(df, query_col="quid", relevance_col="rel", measure=rr)
    quid
    1    1.0
    2    0.5
    Name: rel, dtype: float64
    """
    f = partial(measure, y_pred=None, k=k, relevance=_relevance)
    return df.groupby(query_col)[relevance_col].apply(f)
