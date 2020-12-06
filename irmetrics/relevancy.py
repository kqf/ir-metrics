

def unilabel(y_true, y_pred):
    """Compute relevance(s) of predicted labels.
    This version of the relevancy function works only for the queries
    (problems) with a single groud truth label.

    It is provided mainly for two reasons: there is a slight speedup (order of
    seconds for the large `n_samples`) and it adds expresivity
    if needed.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples, 1), where `n_samples >= 1`
        Ground true labels for a given query (as returned by an IR system).
    y_pred : ndarray of shape (n_samples, n_labels), where `n_samples >= 1`
        Target labels sorted by relevance (as returned by an IR system).
    Returns
    -------
    relevance : bolean ndarray
        The relevance judgements for `y_pred` of shape (n_samples, 1)
    Examples
    --------
    >>> import numpy as np
    >>> from irmetrics.relevancy import unilabel
    >>> # groud-truth label of some answers to a query:
    >>> y_true = np.array([[1]]) # (1, 1)
    >>> # and the predicted labels by an IR system
    >>> y_pred = np.array([[0, 1, 4]]) # (1, 3)
    >>> unilabel(y_true, y_pred)
    array([[False,  True, False]])
    >>> y_true = np.array([[1], [2]]) # (2, 1)
    >>> y_pred = np.array([[0, 1, 4], [5, 6, 7]]) # (2, 3)
    >>> unilabel(y_true, y_pred)
    array([[False,  True, False],
           [False, False, False]])
    >>> # Now the multilabel case:
    >>> y_true = np.array([[1, 4]]) # (1, 2)
    >>> y_pred = np.array([[0, 1, 4]]) # (1, 3)
    >>> unilabel(y_true, y_pred)
    False
    """
    return y_true == y_pred


def multilabel(y_true, y_pred):
    """Compute relevance(s) of predicted labels.
    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_true), where `n_samples >= 1`
        Ground true labels for a given query (as returned by an IR system).
    y_pred : ndarray of shape (n_samples, n_labels), where `n_samples >= 1`
        Target labels sorted by relevance (as returned by an IR system).
        The `n_labels` and `n_true` may not be the same.
    Returns
    -------
    relevance : bolean ndarray
        The relevance judgements for `y_pred` of shape (n_samples, 1)
    Examples
    --------
    >>> import numpy as np
    >>> from irmetrics.relevancy import multilabel
    >>> # groud-truth label of some answers to a query:
    >>> y_true = np.array([[1]]) # (1, 1)
    >>> # and the predicted labels by an IR system
    >>> y_pred = np.array([[0, 1, 4]]) # (1, 3)
    >>> multilabel(y_true, y_pred)
    array([[False,  True, False]])
    >>> y_true = np.array([[1], [2]]) # (2, 1)
    >>> y_pred = np.array([[0, 1, 4], [5, 6, 7]]) # (2, 3)
    >>> multilabel(y_true, y_pred)
    array([[False,  True, False],
           [False, False, False]])
    >>> # Now the multilabel case:
    >>> y_true = np.array([[1, 4]]) # (1, 2)
    >>> y_pred = np.array([[0, 1, 4]]) # (1, 3)
    >>> multilabel(y_true, y_pred)
    array([[False,  True,  True]])
    """
    return (y_pred[:, :, None] == y_true[:, None]).any(axis=-1)
