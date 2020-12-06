def unilabel(y_true, y_pred):
    """Compute relevance(s) of predicted labels.
    This version of the relevance function works only for the queries
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
    Raises
    -------
    ValueError
        If `y_true` has last dimension larger than 1 (multilabel case).
    Examples
    --------
    >>> import numpy as np
    >>> from irmetrics.relevance import unilabel
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
    """
    if y_true.shape[-1] != 1:
        msg = "y_true is expected to be of shape (n_samples, 1), got {}"
        raise ValueError(msg.format(y_true.shape))
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
        The relevance judgements for `y_pred` of shape (n_samples, n_labels)
    Examples
    --------
    >>> import numpy as np
    >>> from irmetrics.relevance import multilabel
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


def relevant_counts(y_pred, y_true):
    """Calculate the total number of relevant items.
    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_true), where `n_samples >= 1`
        Ground true labels for a given query (as returned by an IR system).
    y_pred : ndarray of shape (n_samples, n_labels), where `n_samples >= 1`
        Target labels sorted by relevance (as returned by an IR system).
        The `n_labels` and `n_true` may not be the same.
    Returns
    -------
    relevance_counts: ndarray
        The number of true relevance judgements for `y_pred`.
    Examples
    --------
    >>> import numpy as np
    >>> from irmetrics.relevance import relevant_counts
    >>> # groud-truth label of some answers to a query:
    >>> y_true = np.array([[1]]) # (1, 1)
    >>> # and the predicted labels by an IR system
    >>> y_pred = np.array([[0, 1, 4]]) # (1, 3)
    >>> relevant_counts(y_true, y_pred)
    array([[1]])
    >>> y_true = np.array([[1], [2]]) # (2, 1)
    >>> y_pred = np.array([[0, 1, 4], [5, 6, 7]]) # (2, 3)
    >>> relevant_counts(y_true, y_pred)
    array([[1],
           [1]])
    >>> # Now the relevant_co  unts case:
    >>> y_true = np.array([[1, 4]]) # (1, 2)
    >>> y_pred = np.array([[0, 1, 4]]) # (1, 3)
    >>> relevant_counts(y_true, y_pred)
    array([[1, 1]])
    """
    return (y_pred[:, :, None] == y_pred[:, None]).sum(axis=-1)
