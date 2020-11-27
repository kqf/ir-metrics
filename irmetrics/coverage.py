import numpy as np
from irmetrics.io import to_scalar


def coverage(y_pred, padding=None):
    """Compute Coverage(s)
    Check if ``y_pred`` contains any nontrivial results.

    This ranking metric yields a high value if true labels are ranked high by
    ``y_pred``.
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
    >>> y_pred = [ None]
    >>> coverage(y_true)
    0
    """
    return to_scalar(np.not_equal(y_pred, padding).sum(axis=-1) > 0)
