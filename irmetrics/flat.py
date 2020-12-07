from functools import partial


def _relevance(y_true, y_pred):
    return y_true


def flat(df, query_col, relevance_col, measure, k=None):
    f = partial(measure, y_pred=None, k=k, relevance=_relevance)
    return df.groupby(query_col)[relevance_col].apply(f)
