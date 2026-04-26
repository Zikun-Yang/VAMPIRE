import polars as pl
from typing import List, Dict


def prepare_largest_trs_table_data(df: pl.DataFrame, topn: int = 20) -> List[Dict]:
    """
    prepare data for the largest TRs table by copy number
    Input:
        df: pl.DataFrame
    Output:
        data: list of dicts
    """
    df = (
        df.sort("copyNumber", descending=True)
        .head(topn)
        .select(["chrom", "start", "end", "period", "copyNumber", "percentMatches", "percentIndels", "score", "entropy", "motif"])
    )
    return [dict(zip(df.columns, row)) for row in df.iter_rows()]
