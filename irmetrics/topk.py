import numpy as np


def rr(y_true, y_pred, k=20):
    """Compute Recirocal Rank(s).
    Calculate the recirocal of the index for the first matched item in
    `y_pred`. The score is between 0 and 1.

    This ranking metric yields a high value if true labels are ranked high by
    ``y_pred``.
    Parameters
    ----------
    y_true : scalar, iterable or ndarray of shape (n_samples, n_labels)
        True labels of entities to be ranked. In case of scalars ``y_pred``
        should be of shape (1, n_labels).
    y_pred : iterable, ndarray of shape (n_samples, n_labels)
        Target labels sorted by relevance (as returned by an IR system).
    k : int, default=20
        Only consider the highest k scores in the ranking. If None, use all
        outputs.
    Returns
    -------
    rr : float in [0., 1.]
        The recirocal ranks for all samples.
    References
    ----------
    `Wikipedia entry for Mean reciprocal rank
    <https://en.wikipedia.org/wiki/Mean_reciprocal_rank>`_
    Examples
    --------
    >>> from irmetrics.topk import rr
    >>> # we have groud-truth label of some answers to a query:
    >>> y_true = 1
    >>> # and the predicted labels by an IR system
    >>> y_pred = [0, 1, 4]
    >>> ndcg_score(y_true, y_pred)
    0.5
    """
    y_true, y_pred = np.atleast_2d(y_true, y_pred)
    y_true = y_true.T[:, :k]
    y_pred = y_pred[:, :k]

    relevant = y_true == y_pred
    index = relevant.argmax(-1)
    rrs = np.squeeze(relevant.any(-1) / (index + 1))
    if not rrs.shape:
        return rrs.item()
    return rrs


def recall(y_true, y_pred=None, ignore=None, k=20):
    y_true, y_pred = np.atleast_2d(y_true, y_pred)
    y_true = y_true.T[:, :k]
    y_pred = y_pred[:, :k]

    relevant = (y_true == y_pred).any(-1) / y_true.shape[-1]
    recalls = np.squeeze(relevant)
    if not recalls.shape:
        return recalls.item()
    return recalls
