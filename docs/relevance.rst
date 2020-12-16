=================================
Using custom relevance judgements
=================================

All top-k metrics accept the relevance function as a parameter. This way it is possible to modify the behavior of the metrics.

.. code:: python

    >>> from irmetrics.topk import rr
    >>> from irmetrics.relevance import multilabel
    >>> y_true = "apple"
    >>> y_pred = ["banana", "apple", "grapes", "invalid_output"]
    >>> def relevance(y_true, y_pred):
    ...     return ~multilabel(y_true, y_pred)
    >>> rr(y_true, y_pred, relevance=relevance)
    1.0

Similarly this code can be adapted for inputs with multiple queries.
