from typing import Tuple
import polars as pl

def calculate_window_stats(chrom_df: pl.DataFrame, w_start: int, w_end: int) -> Tuple[float, int, int]:
    """
    Calculate the window stats
    Input:
        chrom_df: pl.DataFrame
        w_start: int
        w_end: int
    Output:
        tr_fraction: float
        median_length: int
        median_period: int
    """
    sub = chrom_df.filter(
        (pl.col("end") >= w_start) &
        (pl.col("start") <= w_end)
    )

    if sub.height == 0:
        return 0.0, None, None

    cum_len: int = 0
    cur_pos: int = w_start
    for row in sub.iter_rows(named=True):
        if row["end"] < cur_pos:
            continue
        elif row["start"] > cur_pos:
            cum_len += row["end"] - row["start"] + 1
        else:
            cum_len += row["end"] - cur_pos + 1
        cur_pos = row["end"] + 1

    tr_fraction = cum_len / (w_end - w_start + 1)

    return (
        float(tr_fraction),
        float(sub["length"].median()),
        float(sub["period"].median())
    )