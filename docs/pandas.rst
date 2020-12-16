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
Quite often data is represented in long (or flat) format and only relevance judgements provided for each entry.
There is a dedicated `irmetrics.flat` module created for that:

.. code:: python

    >>> import numpy as np
    >>> import pandas as pd
    >>> from irmetrics.flat import flat
    >>> # example data
    >>> df = pd.DataFrame({
    ...    "click": [0, 1, 0, 1, 0, 0],
    ...    "label": ["banana", "apple", "grapes", "bob", "rob", "don"],
    ...    "query_id": [0, 0, 0, 1, 1, 1]})
    >>> df
       click   label  query_id
    0      0  banana         0
    1      1   apple         0
    2      0  grapes         0
    3      1     bob         1
    4      0     rob         1
    5      0     don         1
    >>> # calculate the MRR
    >>> flat(df, query_col="query_id", relevance_col="click", measure=rr)
    query_id
    0    0.5
    1    1.0
    Name: click, dtype: float64

In the example above, "label" column is provided just for illustration purposes and is ignored. Currently `ir-metrics` defines only `ndcg` and `rr` measures that are compatible with flat format.
