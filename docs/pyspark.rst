===================
Using with `pandas`
===================

The metrics are designed to work also with `pandas` dataframes:

.. code:: python

    >>> import pandas as pd
    >>> import pyspark.sql.functions as F
    >>> from pyspark import SparkContext
    >>> from pyspark.sql import SQLContext
    >>> from irmetrics.topk import rr
    >>> # basic data
    >>> y_true = "apple"
    >>> y_pred = ["banana", "apple", "grapes"]
    >>> n = 10
    >>> # create the example dataframe by repeating entries n times
    >>> df = pd.DataFrame({
    >>>     "y_true": [y_true] * n,
    >>>     "y_pred": [y_pred] * n
    >>> })
    >>> # Create spark datasets
    >>> spark = SQLContext(SparkContext())
    >>> sdf = spark.createDataFrame(df)
    >>> # apply the metics
    >>> sdf.withColumn("rr", F.udf(rr)("y_true", "y_pred")).show()
    0
