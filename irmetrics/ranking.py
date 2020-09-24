import numpy as np
from functools import partial


def dcg_score(ranking, at=None, weights=1.0):
    target = ranking if at is None else ranking[:at]
    gains = (2 ** target - 1) / np.log2(np.arange(target.size) + 2)
    return np.sum(gains * weights)


def ndcg_score(ranking, at=None):
    ranking = np.array(ranking)
    idcg_score = dcg_score(np.sort(ranking)[::-1], at)
    return dcg_score(ranking, at) / idcg_score


def ndcg_scores(df, query_col, relevance_col, at=None):
    return df.groupby(query_col)[relevance_col].agg(
        lambda x: ndcg_score(x, at))


def mean_ndcg_score(df, query_col, relevance_col, at=10, skipna=None):
    return ndcg_scores(df, query_col, relevance_col, at=at).mean(skipna=skipna)


def weigted_dcg_score(relevance, weights, at=None):
    relevance, weights = relevance[:at], weights[:at]
    gains = (2 ** relevance - 1) / np.log2(np.arange(relevance.size) + 2)
    return np.sum(gains * weights)


def wdcg_score(x, at=None):
    relevance, weights = x.values.T
    idx = relevance.argsort()[::-1]
    idcg_score = weigted_dcg_score(relevance[idx], weights=weights[idx], at=at)
    return weigted_dcg_score(relevance, weights=weights, at=at) / idcg_score


def wdcg_scores(df, query_col, relevance_col, weights_col, at=None):
    wdcgs = df.groupby(query_col)[relevance_col, weights_col].apply(
        partial(wdcg_score, at=at))
    return wdcgs


def mean_wdcg_score(df, query_col, relevance_col, weights_col,
                    at=10, skipna=None):
    return wdcg_scores(df, query_col,
                       relevance_col, weights_col, at=at).mean(skipna=skipna)
