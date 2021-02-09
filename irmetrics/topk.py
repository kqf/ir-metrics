import numpy as np

from irmetrics.io import _ensure_io, _validate_unique
from irmetrics.relevance import multilabel


@_ensure_io
@_validate_unique
def rr(y_true, y_pred, k=None, relevance=multilabel):
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
    k : int, default=None
        Only consider the highest k scores in the ranking. If None, use all
        outputs.
    relevance : callable, default=topk.relevance.multilabel
        A function that calculates relevance judgements based on input
        ``y_pred`` and ``y_true``.

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
    >>> y_true = 1

    and the predicted labels by an IR system:

    >>> y_pred = [0, 1, 4]
    >>> rr(y_true, y_pred)
    0.5
    """
    relevant = relevance(y_true, y_pred)
    index = relevant.argmax(-1)
    return relevant.any(-1) / (index + 1)


@_ensure_io
@_validate_unique
def recall(y_true, y_pred=None, k=None, relevance=multilabel, pad_symbol=None):
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
    k : int, default=None
        Only consider the highest k scores in the ranking. If None, use all
        outputs.
    relevance : callable, default=topk.relevance.multilabel
        A function that calculates relevance judgements based on input
        ``y_pred`` and ``y_true``.
    pad_symbol : callable, default=None
        A value that was used to pad the `y_true`. This is needed to ignore
        the padding when calculating the recall. The default value is `None`,
        this means `numpy` will do element-wise comparison for `None`, change
        it to somehting else if your predictions may contain `None` objects.

    Returns
    -------
    recall : float in [0., 1.]
        The relevances for all samples.

    References
    ----------
    `Wikipedia entry for precision and recall
    <https://en.wikipedia.org/wiki/Precision_and_recall>`_

    Examples
    --------
    >>> from irmetrics.topk import recall

    for ground-truth labels related to a query:

    >>> y_true = 1

    and the predicted labels by an IR system:

    >>> y_pred = [0, 1, 4]
    >>> recall(y_true, y_pred)
    1.0
    """
    positives = ~np.equal(y_true, pad_symbol)
    return relevance(y_true, y_pred).sum(-1) / positives.sum(-1)


@_ensure_io
@_validate_unique
def precision(y_true, y_pred=None, k=None, relevance=multilabel):
    """Compute Recall(s).
    and 1 otherwise.
    Check which fraction of ``y_pred`` is in ``y_true``.
    **NB**: When passing ``y_pred`` of shape `[n_samples, n_outputs]`
    the result is quivalent to `recall(y_pred, y_true) / n_outputs`.

    Parameters
    ----------
    y_true : scalar, iterable or ndarray of shape (n_samples, n_labels)
        True labels of entities to be ranked. In case of scalars ``y_pred``
        should be of shape (1, n_labels).
    y_pred : iterable, ndarray of shape (n_samples, n_labels)
        Target labels sorted by relevance (as returned by an IR system).
    k : int, default=None
        Only consider the highest k scores in the ranking. If None, use all
        outputs.
    relevance : callable, default=topk.relevance.multilabel
        A function that calculates relevance judgements based on input
        ``y_pred`` and ``y_true``.

    Returns
    -------
    precision : float in [0., 1.]
        The relevances for all samples.

    References
    ----------
    `Wikipedia entry for precision and recall
    <https://en.wikipedia.org/wiki/Precision_and_recall>`_

    Examples
    --------
    >>> from irmetrics.topk import recall

    for ground-truth labels related to a query:

    >>> y_true = 1

    and the predicted labels by an IR system:

    >>> y_pred = [0, 1, 4, 3]
    >>> precision(y_true, y_pred)
    0.25
    """
    return relevance(y_true, y_pred).sum(-1) / y_pred.shape[-1]


def dcg_score(relevance, k=None, weights=1.0):
    """Compute Discounted Cumulative Gain score(s) based on `relevance`
    judgements provided.

    This is provided as internal implementation for `ndcg` for this reason
    the API for this function slightly differ: it alawyas accepts  and
    outputs `np.arrays`, unlike other methos in this module.

    Parameters
    ----------
    relevance : iterable or ndarray of shape (n_samples, n_labels) or simply
        (n_labels,). The last dimension of the parameter is used as position.
        The relevance judgements provided by experts.
    weights : default=1.0, scalar, iterable or ndarray of shape (n_samples,)
        takes into account the importance of each sample, if relevant.
    k : int, default=None
        Only consider the highest k scores in the ranking. If None, use all
        outputs.
    relevance : callable, default=topk.relevance.multilabel
        A function that calculates relevance judgements based on input
        ``y_pred`` and ``y_true``.

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
    >>> from irmetrics.topk import dcg_score

    for ground-truth labels related to a query:

    >>> relevance_judgements = np.array([[1, 0, 0, 0]])
    >>> dcg_score(relevance_judgements)
    array([1.])
    >>> relevance_judgements = np.array([[True, False, False, False]])
    >>> dcg_score(relevance_judgements)
    array([1.])
    >>> relevance_judgements = np.array([[False, True, False, False]])
    >>> dcg_score(relevance_judgements)
    array([0.63092975])
    """
    top = relevance[..., :k]
    gains = (2 ** top - 1) / np.log2(np.arange(top.shape[-1]) + 2)[None, ...]
    return np.sum(gains * weights, axis=-1)


@_ensure_io
@_validate_unique
def ndcg(y_true, y_pred, k=None, relevance=multilabel, weights=1.):
    """Compute Normalized Discounted Cumulative Gain score(s) based on
    `relevance` judgements provided.

    Parameters
    ----------
    y_true : iterable or ndarray of shape (n_samples, n_labels) or simply
        (n_labels,). The last dimension of the parameter is used as position.
    y_pred : iterable, ndarray of shape (n_samples, n_labels)
        Target labels sorted by relevance (as returned by an IR system).
    k : int, default=None
        Only consider the highest k scores in the ranking. If None, use all
        outputs.
    weights : float, iterable, ndarray, default=1.0
        Represents the weights of each sample.
    relevance : callable, default=topk.relevance.multilabel
        A function that calculates relevance judgements based on input
        ``y_pred`` and ``y_true``.

    Returns
    -------
    ndcg : np.array
        The discounted cumulative gains for samples (or a single sample).

    References
    ----------
    `Wikipedia entry for normalized discounted cumulative gain
    <https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG>`_

    Examples
    --------
    >>> from irmetrics.topk import ndcg

    for ground-truth labels related to a query:

    >>> y_true = [1, 2]
    >>> y_pred = [0, 1, 0, 0]
    >>> ndcg(y_true, y_pred)
    0.6309297535714575
    >>> # the order of y_true labels doesn't matter
    >>> y_true = [2, 1]
    >>> y_pred = [0, 1, 0, 0]
    >>> ndcg(y_true, y_pred)
    0.6309297535714575
    """
    relevant = relevance(y_true, y_pred)

    # Sort in descending order, calculate the gain
    idcg = dcg_score(np.flip(np.sort(relevant, axis=-1), axis=-1), k, weights)

    # Normalize to the ideal dcg score
    return dcg_score(relevant, k) / idcg


@_ensure_io
@_validate_unique
def ap(y_true, y_pred, k=None, relevance=multilabel):
    """Compute Average Precision score(s).
    AP is an aproximation of the integral over PR-curve.

    Parameters
    ----------
    y_true : scalar, iterable or ndarray of shape (n_samples, n_labels)
        True labels of entities to be ranked. In case of scalars ``y_pred``
        should be of shape (1, n_labels).
    y_pred : iterable, ndarray of shape (n_samples, n_labels)
        Target labels sorted by relevance (as returned by an IR system).
    k : int, default=None
        Only consider the highest k scores in the ranking. If None, use all
        outputs. The minimum between the nuber of correct answers and k will
        be used to compute the score.
    relevance : callable, default=topk.relevance.multilabel
        A function that calculates relevance judgements based on input
        ``y_pred`` and ``y_true``.

    Returns
    -------
    ap : float
        The average precision for a given sample.

    References
    ----------
    `Wikipedia entry for Mean Average Precision
    <https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision>`_

    Examples
    --------
    >>> from irmetrics.topk import ap

    for ground-truth labels related to a query:

    >>> y_true = 1

    and the predicted labels by an IR system:

    >>> y_pred = [1, 0, 0]
    >>> ap(y_true, y_pred)
    0.3333333333333333
    >>> # This should be fixed
    >>> y_true = [1, 4, 5]

    and the predicted labels by an IR system:

    >>> y_pred = [1, 2, 3, 4, 5]
    >>> ap(y_true, y_pred)
    array([0.2, 0. , 0. ])
    """
    relevant = relevance(y_true, y_pred)

    # Handle k=None, without if else branching
    max_iter = min(i for i in (k, y_true.shape[-1]) if i is not None)

    ap = np.sum([
        np.array(
            precision(y_true, y_pred, ik + 1)
        )[..., None] * relevant[..., [ik]]
        for ik in range(max_iter)
    ], axis=-1)

    return ap / y_pred.shape[-1]
