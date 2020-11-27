import numpy as np
from functools import wraps


def _ensure_io(f):
    @wraps(f)
    def wrapper(y_true, y_pred, k=20):
        # Ensure (n_samples, n_labels) shapes for the inputs
        y_true, y_pred = np.atleast_2d(y_true, y_pred)

        # Take at most k labels
        y_true = y_true.T[:, :k]
        y_pred = y_pred[:, :k]

        # Calculate the measure
        raw_outputs = f(y_true, y_pred, k)

        # Remove unwanted dimensions if any
        outputs = np.squeeze(raw_outputs)
        if not outputs.shape:
            return outputs.item()
        return outputs

    return wrapper


@_ensure_io
def rr(y_true, y_pred, k=20):
    """Compute Recirocal Rank(s).
    Calculate the recirocal of the index for the first matched item in
    ``y_pred``. The score is between 0 and 1.

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
    >>> rr(y_true, y_pred)
    0.5
    """
    relevant = y_true == y_pred
    index = relevant.argmax(-1)
    return relevant.any(-1) / (index + 1)


@_ensure_io
def recall(y_true, y_pred=None, ignore=None, k=20):
    """Compute Recall(s).
    Check if at least one metric proposed in ``y_pred`` is in ``y_true``.
    This is the binary score, 0 -- all predictionss are irrelevant
    and 1 otherwise.
    This definition of recall is equivalent to accuracy@k.
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
    rr : bool in [True, False]
        The relevances for all samples.
    References
    ----------
    `Wikipedia entry for precision and recall
    <https://en.wikipedia.org/wiki/Precision_and_recall>`_
    Examples
    --------
    >>> from irmetrics.topk import recall
    >>> # we have groud-truth label of some answers to a query:
    >>> y_true = 1
    >>> # and the predicted labels by an IR system
    >>> y_pred = [0, 1, 4]
    >>> recall(y_true, y_pred)
    1.0
    """
    return (y_true == y_pred).any(-1) / y_true.shape[-1]


@_ensure_io
def precision(y_true, y_pred=None, ignore=None, k=20):
    """Compute Recall(s).
    and 1 otherwise.
    Check which fraction of ``y_pred`` is in ``y_true``.
    **NB**: When passing ``y_pred` of shape `[n_samples, n_outputs]`
    the result is quivalent to `recall(y_pred, y_true) / n_outputs`.
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
    rr : bool in [True, False]
        The relevances for all samples.
    References
    ----------
    `Wikipedia entry for precision and recall
    <https://en.wikipedia.org/wiki/Precision_and_recall>`_
    Examples
    --------
    >>> from irmetrics.topk import recall
    >>> # we have groud-truth label of some answers to a query:
    >>> y_true = 1
    >>> # and the predicted labels by an IR system
    >>> y_pred = [0, 1, 4, 3]
    >>> precision(y_true, y_pred)
    0.25
    """
    return (y_true == y_pred).any(-1) / y_pred.shape[-1]


def dcg_score(relevancy, k=None, weights=1.0):
    """Compute Discounted Cumulative Gain score(s) based on `relevancy`
    judgements provided.
    Parameters
    ----------
    relevancy : iterable or ndarray of shape (n_samples, n_labels) or simply
        (n_labels,). The last dimension of the parameter is used as position.
    weights : default=1.0, scalar, iterable or ndarray of shape (n_samples,)
        takes into account the importance of each sample, if relevant.
    k : int, default=20
        Only consider the highest k scores in the ranking. If None, use all
        outputs.
    Returns
    -------
    dcg : np.array
        The discounted cumulative gains for samples (or a single sample).
    References
    ----------
    `Wikipedia entry for Discounted cumulative gain
    <https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain>`_
    Examples
    --------
    >>> from irmetrics.topk ort dcg_score
    >>> # we have groud-truth label of some answers to a query:
    >>> relevancy_judgements = [1, 0, 0, 0]
    >>> dcg_score(relevancy_judgements)
    1.0
    """
    top = relevancy[..., :k]
    gains = (2 ** top - 1) / np.log2(np.arange(top.shape[-1]) + 2)[None, ...]
    return np.sum(gains * weights, axis=-1)


def ndcg_score(relevant, k=25, weights=1.0):
    """Compute Normalized Discounted Cumulative Gain score(s) based on
    `relevancy` judgements provided.
    Parameters
    ----------
    relevancy : iterable or ndarray of shape (n_samples, n_labels) or simply
        (n_labels,). The last dimension of the parameter is used as position.
    y_pred : iterable, ndarray of shape (n_samples, n_labels)
    k : int, default=20
        Only consider the highest k scores in the ranking. If None, use all
        outputs.
    Returns
    -------
    ndcg : np.array
        The discounted cumulative gains for samples (or a single sample).
    References
    ----------
    `Wikipedia entry for Discounted cumulative gain
    <https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG>`_
    Examples
    --------
    >>> from irmetrics.topk ort dcg_score
    >>> # we have groud-truth label of some answers to a query:
    >>> relevancy_judgements = [1, 0, 0, 0]
    >>> ndcg_score(relevancy_judgements)
    1.0
    """
    # Sort in descending order, calculate the gain
    idcg = dcg_score(np.flip(np.sort(relevant, axis=-1), axis=-1), k, weights)

    # Normalize to the ideal dcg score
    return dcg_score(relevant, k) / idcg


@_ensure_io
def ndcg(y_true, y_pred, k=25):
    """Compute Discounted Cumulative Gain score(s) based on `relevancy`
    judgements provided.
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
    ndcg : np.array
        The discounted cumulative gains for samples (or a single sample).
    References
    ----------
    `Wikipedia entry for Discounted cumulative gain
    <https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG>`_
    Examples
    --------
    >>> from irmetrics.topk ort dcg_score
    >>> # we have groud-truth label of some answers to a query:
    >>> y_pred = [1, 0, 0, 0]
    >>> ndcg(y_true, y_pred)
    1.0
    """
    relevant = (y_pred[:, :, None] == y_true[:, None]).any(axis=-1)
    return ndcg_score(relevant, k)
