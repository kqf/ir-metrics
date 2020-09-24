import numpy as np


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
