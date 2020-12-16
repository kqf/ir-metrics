===================
Using with `pandas`
===================

The metrics are designed to work also with `pandas` dataframes:

.. code:: python

    >>> import numpy as np
    >>> import pandas as pd
    >>> from irmetrics.topk import rr
    >>> # basic data
    >>> y_true = "apple"
    >>> y_pred = ["banana", "apple", "grapes"]
    >>> n = 10
    >>> # create the example dataframe by repeating entries n times
    >>> df = pd.DataFrame({"y_true": [y_true] * n, "y_pred": [y_pred] * n})
    >>> # calculate the MRR
    >>> rr(df["y_true"], np.vstack(df["y_pred"])).mean()
    0.5

Note that `np.vstack` is required here to convert `y_pred` to matrix.
