import sys
import shlex
import polars as pl
from pathlib import Path
from Bio import SeqIO
from typing import Tuple, List, Dict
import logging

from ._motif_processing import canonicalize_motif

logger = logging.getLogger(__name__)

def format_time(time_used: float) -> str:
    """
    Format the time used. Return in the format of "X s", "X min", "X h".
    Input:
        time_used: float
    Output:
        time: str
    """
    time_used: float = time_used
    if time_used < 60:
        return f"{time_used:.2f} s"
    elif time_used < 3600:
        return f"{time_used / 60:.2f} min"
    else:
        return f"{time_used / 3600:.2f} h"

def get_full_command_line():
    """
    Get the full command line of the script
    Input:
        None
    Output:
        command: str
    """
    cmd_parts = [shlex.quote(arg) for arg in sys.argv]
    return ' '.join(cmd_parts)

def get_representative_motifs(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get the representative motifs
    Input:
        df: pl.DataFrame
    Output:
        df: pl.DataFrame
    """
   
    df = df.with_columns(pl.col("motif").map_elements(canonicalize_motif).alias("canonical_motif"))
    return df

def make_fasta_metainfo(fasta_filepath: str) -> dict:
    """
    Make the fasta metainfo
    Input:
        fasta_filepath: str
    Output:
        metainfo: dict, format: {chrom_name: length}
    """
    metainfo: dict[str, int] = {} # name -> length
    # read fasta file
    for record in SeqIO.parse(fasta_filepath, "fasta"):
        metainfo[record.id] = len(record.seq)
    return metainfo

def calculate_smoothness_coverage(smoothness: Dict[int, pl.DataFrame], min_smoothness: int) -> str:
    """
    Get the smoothness coverage
    Input:
        smoothness: Dict[int, pl.DataFrame]
        min_smoothness: int
    Output:
        coverage: str, e.g. "1,000"
    """
    table_raw_template = """
    <tr>
        <td class="table-key">$KEY$</td>
        <td class="table-value">$VALUE$</td>
    </tr>
    """
    table_content = ""
    ksize_list: List[int] = sorted(list(smoothness.keys()))

    # iterate over files, each k size has a trace
    for ksize in ksize_list:
        data = smoothness[ksize]
        if "count" in data.columns and data.height > 0:
            base_count = data.select(pl.col("count").sum()).item()
            base_beyond_threshold_count = data.filter(pl.col("score") > min_smoothness).select(pl.col("count").sum()).item()
            coverage = float(base_beyond_threshold_count) / float(base_count)
        else:
            coverage = 0
        table_content += table_raw_template.replace("$KEY$", f"Smoothness above threshold (k={ksize})").replace("$VALUE$", f"{coverage:.2%}")

    return f"""
    <table class="table">
    <tbody>
    {table_content}
    </tbody>
    </table>
    """

def get_number_of_raw_trs(job_dir: str) -> int:
    """
    Get the number of raw trs
    Input:
        job_dir: str
    Output:
        number: str, e.g. "1,000"
    """
    number = 0
    filepaths = list(Path(job_dir).glob("raw_rgns/*.tsv"))
    for filepath in filepaths:
        if filepath.stat().st_size == 0:
            logger.debug(f"Skipping empty file: {filepath}")
            continue
        lf = pl.scan_csv(filepath, separator="\t", has_header=False, infer_schema_length=0)
        n_rows = lf.select(pl.len()).collect().item()
        number += n_rows
    return f"{number:,}"

def get_number_of_polished_trs(job_dir: str) -> int:
    """
    Get the number of polished trs
    Input:
        job_dir: str
    Output:
        number: str, e.g. "1,000"
    """
    number = 0
    filepaths = list(Path(job_dir).glob("polished_rgns/*.tsv"))
    for filepath in filepaths:
        if filepath.stat().st_size == 0:
            logger.debug(f"Skipping empty file: {filepath}")
            continue
        lf = pl.scan_csv(filepath, separator="\t", has_header=False, infer_schema_length=0)
        n_rows = lf.select(pl.len()).collect().item()
        number += n_rows
    return f"{number:,}"

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
        else:
            cum_len += min(row["end"], w_end) - max(row["start"], cur_pos) + 1
        cur_pos = row["end"] + 1

    tr_fraction = cum_len / (w_end - w_start + 1)

    return (
        float(tr_fraction),
        float(sub["length"].median()),
        float(sub["period"].median())
    )