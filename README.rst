ir-metrics |build| |downloads|
------------------------------

.. |build| image:: https://travis-ci.com/kqf/ir-metrics.svg?branch=master
    :alt: Build Status
    :scale: 100%
    :target: https://travis-ci.com/kqf/ir-metrics

.. |downloads| image:: https://img.shields.io/pypi/dm/ir-metrics
    :alt: PyPi downloads
    :scale: 100%
    :target: https://img.shields.io/pypi/dm/ir-metrics

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

============
Installation
============

To install with pip, run:

.. code:: bash

    pip install ir-metrics
