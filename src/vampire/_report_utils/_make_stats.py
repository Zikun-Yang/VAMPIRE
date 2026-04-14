import logging
from pathlib import Path
from datetime import datetime
from importlib.metadata import version
from typing import Dict
import polars as pl

# plotting
import plotly.graph_objects as go

# custom functions
from ._read import (
    read_tr_file,
    read_smoothness_file
)
from ._stats_basic import (
    format_time, 
    get_full_command_line, 
    get_representative_motifs, 
    make_fasta_metainfo, 
    get_number_of_raw_trs, 
    get_number_of_polished_trs, 
    calculate_smoothness_coverage
)
from ._plot import (
    plot_smoothness_score_distribution, 
    plot_tr_distribution, 
    plot_length_period_distribution, 
    plot_entropy_distribution, 
    plot_enriched_motifs_by_tr_number, 
    plot_enriched_motifs_by_copy_number
)
from ._table import (
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

    match params["subcommand"]:
        case "scan":
            # read tr annotation results
            df = read_tr_file(f"{params['job_dir']}/final_results.tsv")
            df = get_representative_motifs(df)

            # read smoothness score (get all smoothness_distribution_*.txt files)
            stats_path = Path(params["job_dir"]) / "stats"
            file_list: list[Path] = list(stats_path.glob("smoothness_distribution_*.txt"))
            ksize_list: list[int] = [int(file.stem.split("_")[-1]) for file in file_list]
            file_ksize_tuple = sorted(zip(file_list, ksize_list), key=lambda x: x[1])
            if not file_ksize_tuple:
                raise FileNotFoundError(f"No smoothness files found in {stats_path}")
            smoothness: Dict[int, pl.DataFrame] = {}
            for file, ksize in file_ksize_tuple:
                smoothness[ksize] = read_smoothness_file(file)
            logger.debug(f"Read and processed smoothness score")

            # make fasta metainfo
            fasta_metainfo = make_fasta_metainfo(params["input"])
            logger.debug(f"Made fasta metainfo")
            scan_extra = {
                "KSIZES": ", ".join(map(str, params["ksize"])),
                "MAX_PERIOD": params["max_period"],
                "ROLLING_WINDOW_SIZE": params["rolling_win_size"],
                "MIN_SMOOTHNESS_SCORE": params["min_smoothness"],
                "ALIGNMENT_PARAMETERS": f"{params['match_score']}, {params['mismatch_penalty']}, {params['gap_open_penalty']}, {params['gap_extend_penalty']}",
                "MIN_ALIGNMENT_SCORE": params["min_score"],
                "MIN_COPY_NUMBER": params["min_copy"],
                "NUMBER_OF_RAW_TRS": get_number_of_raw_trs(params["job_dir"]),
                "NUMBER_OF_POLISHED_TRS": get_number_of_polished_trs(params["job_dir"]),
                "NUMBER_OF_FINAL_TRS": f"{df.height:,}",
                "SMOOTHNESS_PLOT": fig_to_html(plot_smoothness_score_distribution(smoothness)),
                "SMOOTHNESS_COVERAGE": calculate_smoothness_coverage(smoothness, params["min_smoothness"]),
                "TR_DISTRIBUTION_PLOT": fig_to_html(plot_tr_distribution(df, fasta_metainfo)),
                "LENGTH_PERIOD_DISTRIBUTION_PLOT": fig_to_html(plot_length_period_distribution(df)),
                "ENTROPY_DISTRIBUTION_PLOT": fig_to_html(plot_entropy_distribution(df)),
                "ENRICHED_MOTIFS_BY_TR_NUMBER_PLOT": fig_to_html(plot_enriched_motifs_by_tr_number(df)),
                "ENRICHED_MOTIFS_BY_COPY_NUMBER_PLOT": fig_to_html(plot_enriched_motifs_by_copy_number(df)),
                "LARGEST_TRS_BY_COPY_NUMBER_PLOT": fig_to_html(create_largest_trs_table_by_copy_number(df)),
            }
            data.update(scan_extra)
            logger.debug(f"Made scan extra statistics")

        case "anno":
            import vampire as vp
            adata = vp.pp.read_anno(f"{params['prefix']}.annotation.tsv")
            copy_number_list: List[float] = adata.obs["copy_number"]
            anno_extra = {
                "AUTO_MODE": params["auto"],
                "KSIZE": params["ksize"],
                "ALIGNMENT_PARAMETERS": f"{params['match_score']}, {params['mismatch_penalty']}, {params['gap_open_penalty']}, {params['gap_extend_penalty']}",
                "MIN_ALIGNMENT_SCORE": params["min_score"],
                "NUMBER_OF_SAMPLES": adata.obs.shape[0],
                "NUMBER_OF_MOTIFS": adata.var.shape[0],
                "MIN_COPY_NUMBER": copy_number_list.min(),
                "MEAN_COPY_NUMBER": round(copy_number_list.mean(), 1),
                "MEDIAN_COPY_NUMBER": copy_number_list.median(),
                "MAX_COPY_NUMBER": copy_number_list.max(),
                "WATERFALL_PLOT": "to do...",
            }
            data.update(anno_extra)
            logger.debug(f"Made anno extra statistics")

        case _:
            raise ValueError(f"Invalid subcommand: {params['subcommand']}")
    return data

def fig_to_html(fig: go.Figure) -> str:
    """
    Convert a figure to an HTML string
    Input:
        fig: go.Figure
    Output:
        html_div: str
    """
    fig.update_layout(
        font=dict(
            family="Arial",
            size=14,
            color="black"
        )
    )
    html_div = fig.to_html(full_html=False,
                       include_plotlyjs='cdn',
                       config={"displayModeBar": True,
                               "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d"],
                               'toImageButtonOptions': {
                                    'format': 'svg', # one of png, svg, jpeg, webp
                                    'filename': 'report_figure',
                                    'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
                                },
                               "displaylogo": False,
                               "responsive": True})
    return html_div