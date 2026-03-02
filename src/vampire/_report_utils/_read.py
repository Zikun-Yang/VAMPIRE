from typing import List, Dict
import polars as pl

def read_tr_file(filepath: str) -> pl.DataFrame:
    """
    Read a tr.tsv file and convert to a Polars DataFrame
    Input:
        filepath: str
    Output:
        df: pl.DataFrame
    """
    df = pl.read_csv(filepath, separator="\t", has_header=True)

    # convert columns
    df = df.with_columns([
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64),
        pl.col("period").cast(pl.Float64),
        pl.col("copyNumber").cast(pl.Float64),
    ])

    return df

def read_smoothness_file(file_path: str) -> pl.DataFrame:
    """
    Read a smoothness_distribution_*.txt file and convert to a Polars DataFrame
    with columns: score, count
    """
    # multiple lines
    counts: List[int] = []
    with open(file_path, "r") as f:
        lines: List[str] = f.readlines()
        for line in lines:
            if not counts:
                counts = [int(x) for x in line.split("\t")]  # split by tabs and convert to int
            else:
                ll: List[str] = line.split("\t")
                for i in range(len(ll)):
                    counts[i] += int(ll[i])

    scores = list(range(len(counts)))

    df = pl.DataFrame({
        "score": scores,
        "count": counts
    })

    return df