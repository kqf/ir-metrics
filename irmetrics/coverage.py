import numpy as np
from irmetrics.io import to_scalar, _ensure_io
from irmetrics.relevance import relevant_counts


def coverage(y_pred, padding=None):
    """Compute Coverage(s)
    Check if ``y_pred`` contains any nontrivial results.
    Parameters
    ----------
    y_pred : iterable, ndarray of shape (n_samples, n_labels)
        Target labels sorted by relevance (as returned by an IR system).
    padding : scalar, str, default=None
        The value that was used to pad the predictions to get the same length.
    Returns
    -------
    coverage : int in [0, 1]
        The coverage is 1 if ``y_pred`` contains any results different from
                ``padding`` and 0 otherwise.
    Examples
    --------
    >>> from irmetrics.topk import rr
    >>> # we have groud-truth label of some answers to a query:
    >>> y_true = 1
    >>> # and the predicted labels by an IR system
    >>> y_pred = [0, 1, 4]
    >>> coverage(y_true)
    1
    >>> y_pred = [0, None]
    >>> coverage(y_true)
    1
    >>> y_pred = [None]
    >>> coverage(y_true)
    0
    """
    return to_scalar(np.not_equal(y_pred, padding).sum(axis=-1) > 0)


@_ensure_io
def iou(y_true, y_pred, k=20, relevance=None, n_uniq=relevant_counts):
    """Compute the approximate version of Intersection over Union.
    The approximation comes in assumption that `y_true` and `y_pred`
    contain only unique values.
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
    iou : float in [0., 1.]
        The ratio of relevant retrieved entries to the union of relevant
        and retrieved entries.
    References
    ----------
    `Wikipedia entry for Jaccard Index
    <https://en.wikipedia.org/wiki/Jaccard_index>`_
    Examples
    --------
    >>> from irmetrics.topk import rr
    >>> # we have groud-truth label of some answers to a query:
    >>> y_true = 1
    >>> # and the predicted labels by an IR system
    >>> y_pred = [0, 1, 4]
    >>> iou(y_true, y_pred)
    1. / 3.
    """

    if np.any(n_uniq(y_pred, y_true) > 1):
        raise ValueError("y_pred has duplicates along the last axis")

    if np.any(n_uniq(y_true, y_true) > 1):
        raise ValueError("y_true has duplicates along the last axis")

    relevant = relevance(y_pred, y_true)

    # Approximate intersection
    intersection = (relevant).sum(axis=-1)

    # Approximate union
    union = (y_pred.shape[-1] + y_true.shape[-1] - intersection)

    return intersection / union
