=================================
Using custom relevance judgements
=================================

All top-k metrics accept the relevance function as a parameter. This way it is possible to modify the behavior of the metrics.
In case if each query has only a single positive label one can use `irmetrics.relevance.unilabel` to be more expressive:

.. code:: python

    >>> from irmetrics.topk import rr
    >>> from irmetrics.relevance import unilabel
    >>> y_true = "apple"
    >>> y_pred = ["banana", "apple", "grapes"]
    >>> rr(y_true, y_pred, relevance=unilabel)
    0.5

This gives the same results as the default relevance function but is a (tiny) bit faster.
Similarly, this mechanism allows adding arbitrary logic to the evaluation:

.. code:: python

    >>> from irmetrics.topk import rr
    >>> from irmetrics.relevance import multilabel
    >>> y_true = "apple"
    >>> y_pred = ["banana", "apple", "grapes"]
    >>> def irrelevance(y_true, y_pred):
    ...     return ~multilabel(y_true, y_pred)
    >>> rr(y_true, y_pred, relevance=irrelevance)
    1.0

Similarly this code can be adapted for inputs with multiple queries.
