import polars as pl
from pathlib import Path
from typing import List, Tuple
import plotly.subplots as sp
import plotly.graph_objects as go
from importlib.metadata import version
from datetime import datetime
import logging
from .motif_processing import canonicalize_motif
from . import (
    fig_to_html,
    plot_smoothness_score_distribution,
    plot_tr_distribution,
    plot_length_period_distribution,
    plot_entropy_distribution,
    plot_enriched_motifs_by_tr_number,
    plot_enriched_motifs_by_copy_number,
    create_largest_trs_table_by_copy_number
)

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
        "INPUT": params["input"],
        "OUTPUT": params["prefix"],
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
        """
        if params["composite"]:
            n_composite = df.filter(pl.col("motif").str.contains(",")).height
        else:
            n_composite = 0
        """
        logger.debug(f"Read and processed tr annotation results")
        # make fasta metainfo
        fasta_metainfo = make_fasta_metainfo(params["input"])
        logger.debug(f"Made fasta metainfo")
        scan_extra = {
            "KSIZES": ", ".join(map(str, params["ksize"])),
            "MAX_PERIOD": params["max_period"],
            "ROLLING_WINDOW_SIZE": params["rolling_win_size"],
            "MIN_SMOOTHNESS_SCORE": params["min_smoothness"],
            ### "COMPOSITE": params["composite"],
            "ALIGNMENT_PARAMETERS": f"{params['match_score']}, {params['mismatch_penalty']}, {params['gap_open_penalty']}, {params['gap_extend_penalty']}",
            "MIN_ALIGNMENT_SCORE": params["min_score"],
            "NUMBER_OF_RAW_TRS": _get_number_of_raw_trs(params["job_dir"]),
            "NUMBER_OF_POLISHED_TRS": _get_number_of_polished_trs(params["job_dir"]),
            ### "NUMBER_OF_COMPOSITE_TRS": f"{n_composite:,}",
            "NUMBER_OF_FINAL_TRS": f"{df.height:,}",
            "SMOOTHNESS_PLOT": fig_to_html(plot_smoothness_score_distribution(params["job_dir"])),
            "SMOOTHNESS_COVERAGE": _calculate_smoothness_coverage(params["job_dir"], params["min_smoothness"]),
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
    import Bio.SeqIO
    metainfo: dict[str, int] = {} # name -> length
    # read fasta file
    for record in Bio.SeqIO.parse(fasta_filepath, "fasta"):
        metainfo[record.id] = len(record.seq)
    return metainfo

def _calculate_smoothness_coverage(job_dir: str, min_smoothness: int) -> str:
    """
    Get the smoothness coverage
    Input:
        job_dir: str
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
    
    # get all smoothness_distribution_*.txt files
    stats_path = Path(job_dir) / "stats"
    file_list: list[Path] = list(stats_path.glob("smoothness_distribution_*.txt"))
    ksize_list: list[int] = [int(file.stem.split("_")[-1]) for file in file_list]

    # sort by ksize
    file_ksize_tuple = sorted(zip(file_list, ksize_list), key=lambda x: x[1])
    
    # raise error if no files found
    if not file_ksize_tuple:
        raise FileNotFoundError(f"No smoothness files found in {stats_path}")

    # iterate over files, each k size has a trace
    for file, ksize in file_ksize_tuple:
        data = read_smoothness_file(file)
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

def _get_number_of_raw_trs(job_dir: str) -> int:
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
        lf = pl.scan_csv(filepath, separator="\t", has_header=False)
        n_rows = lf.collect().height
        number += n_rows
    return f"{number:,}"

def _get_number_of_polished_trs(job_dir: str) -> int:
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
        lf = pl.scan_csv(filepath, separator="\t", has_header=False)
        n_rows = lf.collect().height
        number += n_rows
    return f"{number:,}"