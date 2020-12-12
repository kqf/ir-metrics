ir-metrics |build| |downloads| |codecov|
----------------------------------------

.. |build| image:: https://travis-ci.com/kqf/ir-metrics.svg?branch=master
    :alt: Build Status
    :scale: 100%
    :target: https://travis-ci.com/kqf/ir-metrics

.. |downloads| image:: https://img.shields.io/pypi/dm/ir-metrics
    :alt: PyPi downloads
    :scale: 100%
    :target: https://img.shields.io/pypi/dm/ir-metrics

.. |codecov| image:: https://codecov.io/gh/kqf/ir-metrics/branch/master/graph/badge.svg?token=Ow1NHryfq8
    :alt: Code Coverage
    :scale: 100%
    :target: https://codecov.io/gh/kqf/ir-metrics

A set of the most common metrics in used in information retrieval.

============
Usage
============

The metrics are designed to work for array-like structures and integers:

.. code:: python

    >>> from irmetrics.topk import rr
    >>> y_true = "apple"
    >>> y_pred = ["banana", "apple", "grapes"]
    >>> rr(y_true, y_pred)
    0.5

The same function works also for the matrix-like structures:

.. code:: python

    >>> import numpy as np
    >>> from irmetrics.topk import rr
    >>> y_trues = np.repeat(y_true, 128)
    >>> y_preds = np.repeat([y_pred], 128, axis=0)
    >>> # Calculate the Mean Reciprocal Rank
    >>> rr(y_trues, y_preds).mean()
    0.5
    >>> # Calculate the standard deviation for Reciprocal Ranks
    >>> rr(y_trues, y_preds).std()
    0.0

============
Installation
============

To install with pip, run:

.. code:: bash

    pip install ir-metrics
