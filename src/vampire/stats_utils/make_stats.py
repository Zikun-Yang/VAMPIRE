import polars as pl
from pathlib import Path
from typing import List, Tuple
import plotly.subplots as sp
import plotly.graph_objects as go
from importlib.metadata import version
from datetime import datetime
import logging
from vampire.plot_utils import (
    fig_to_html,
    plot_smoothness_score_distribution,
    plot_tr_distribution,
    plot_length_period_distribution,
    plot_entropy_distribution,
    plot_enriched_motifs_by_tr_number,
    plot_enriched_motifs_by_copy_number,
    create_largest_trs_table_by_copy_number)

logger = logging.getLogger(__name__)

def make_stats(params: dict) -> dict:
    """
    make stats
    Input:
        params: dict
    Output:
        data: dict
    """
    data: dict[str, object] = {
        "SUBCMD": params["subcommand"],
        "COMMAND": get_full_command_line(),
        "VERSION": version("vampire-tr"),
        "JOB_NAME": params["job_dir"],
        "TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "TOTAL_RUNNING_TIME": format_time(params["time_used"]),
    }
    logger.debug(f"Collected basic information")

    if params["subcommand"] == "scan":
        # read tr annotation results
        df = read_tr_file(f"{params['job_dir']}/final_results.tsv")
        df = get_representative_motifs(df)
        logger.debug(f"Read and processed tr annotation results")
        # make fasta metainfo
        fasta_metainfo = make_fasta_metainfo(params["fasta"])
        logger.debug(f"Made fasta metainfo")
        scan_extra = {
            "MAX_PERIOD": params["max_period"],
            "MIN_SMOOTHNESS_SCORE": params["min_smoothness"],
            "ROLLING_WINDOW_SIZE": params["rolling_win_size"],
            "SMOOTHNESS_PLOT": fig_to_html(plot_smoothness_score_distribution(params["job_dir"])),
            "TR_DISTRIBUTION_PLOT": fig_to_html(plot_tr_distribution(df, fasta_metainfo)),
            "LENGTH_PERIOD_DISTRIBUTION_PLOT": fig_to_html(plot_length_period_distribution(df)),
            "ENTROPY_DISTRIBUTION_PLOT": fig_to_html(plot_entropy_distribution(df)),
            "ENRICHED_MOTIFS_BY_TR_NUMBER_PLOT": fig_to_html(plot_enriched_motifs_by_tr_number(df)),
            "ENRICHED_MOTIFS_BY_COPY_NUMBER_PLOT": fig_to_html(plot_enriched_motifs_by_copy_number(df)),
            "LARGEST_TRS_BY_COPY_NUMBER_PLOT": fig_to_html(create_largest_trs_table_by_copy_number(df)),
        }
        data.update(scan_extra)
        logger.debug(f"Made scan extra statistics")
    return data

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
    import sys
    import shlex
    cmd_parts = [shlex.quote(arg) for arg in sys.argv]
    return ' '.join(cmd_parts)

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

def get_representative_motifs(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get the representative motifs
    Input:
        df: pl.DataFrame
    Output:
        df: pl.DataFrame
    """
    def _booth_canonicalize(motif: str) -> str:
        """
        Canonicalize motif using Booth algorithm (O(n)).
        Returns the lexicographically smallest cyclic rotation.
        """
        if not motif:
            return motif

        s = motif * 2
        n = len(motif)

        i, j, k = 0, 1, 0
        while i < n and j < n and k < n:
            if s[i + k] == s[j + k]:
                k += 1
            elif s[i + k] > s[j + k]:
                i = i + k + 1
                if i <= j:
                    i = j + 1
                k = 0
            else:
                j = j + k + 1
                if j <= i:
                    j = i + 1
                k = 0

        start = min(i, j)
        return s[start:start + n]
    
    df = df.with_columns(pl.col("motif").map_elements(_booth_canonicalize).alias("canonical_motif"))
    return df

def make_fasta_metainfo(fasta_filepath: str) -> dict:
    """
    Make the fasta metainfo
    Input:
        fasta_filepath: str
    Output:
        metainfo: dict, format: {chrom_name: length}
    """
    import Bio.SeqIO
    metainfo: dict[str, int] = {} # name -> length
    # read fasta file
    for record in Bio.SeqIO.parse(fasta_filepath, "fasta"):
        metainfo[record.id] = len(record.seq)
    return metainfo

