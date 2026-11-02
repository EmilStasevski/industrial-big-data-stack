"""PySpark implementation of the heavy pipeline stages (silver and gold features).

Same outputs as the Polars pipeline (checked by `ruhrwatt spark parity`), written so the
work scales out: one scan of all bronze files, a pivot instead of 25 joins, window functions
for lags and rolling statistics, Hive-partitioned output.
"""
