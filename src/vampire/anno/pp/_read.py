from __future__ import annotations
from typing import TYPE_CHECKING
import polars as pl
import logging
import numba
import numpy as np

from vampire._anno import make_raw

if TYPE_CHECKING:
    import warnings
    import pyarrow
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import anndata as ad
    from vampire._report_utils import(
        get_copy_number
    )

logger = logging.getLogger(__name__)

MATCH_SCORE = 2
MISMATCH_PENALTY = 4
GAP_OPEN_PENALTY = 7
GAP_EXTEND_PENALTY = 4

###
#
# read bed, bedgraph functions
#
###
BEDGRAPH_COLS = {
    "chrom": pl.Utf8,
    "start": pl.Int64,
    "end": pl.Int64,
    "value": pl.Float64,
}

BED_COLS = {
    "chrom": pl.Utf8,
    "start": pl.Int64,
    "end": pl.Int64,
    "name": pl.Utf8,
    "score": pl.Float64,
    "strand": pl.Utf8,
    "thickStart": pl.Int64,
    "thickEnd": pl.Int64,
    "itemRgb": pl.Utf8,
    "blockCount": pl.Int64,
    "blockSizes": pl.Utf8,
    "blockStarts": pl.Utf8,
}

def read_bedgraph(
    bedgraph_file: str,
    columns: dict[str, pl.DataType] = BEDGRAPH_COLS,
) -> pl.LazyFrame:
    """
    Lazily read a BEDGraph file using Polars.

    This function returns a :class:`polars.LazyFrame` and does **not**
    read or parse the file immediately. The file will only be read
    when the lazy query is executed (e.g. via ``collect()``).

    The input file is assumed to be in BEDGraph format with **at least
    four columns**:

    - ``chrom`` (chromosome name)
    - ``start`` (0-based start position)
    - ``end`` (end position, exclusive)
    - ``value`` (value)

    Any additional columns present in the file are preserved and
    automatically assigned standard BEDGraph column names when possible.

    Parameters
    ----------
    bedgraph_file : str
        Path to the input BEDGraph file. Both uncompressed ``.bedgraph`` and
        gzip-compressed ``.bedgraph.gz`` files are supported.

    columns : dict[str, pl.DataType]
        Column names and data types to read.

    Returns
    -------
    polars.LazyFrame
        A lazily-evaluated Polars LazyFrame representing the BEDGraph file
        contents.

    Raises
    ------
    FileNotFoundError
        If the specified BEDGraph file does not exist.
    """
    import polars as pl
    from pathlib import Path
    import warnings

    # check suffix
    if not bedgraph_file.endswith(".bedgraph") and not bedgraph_file.endswith(".bedgraph.gz"):
        warnings.warn(f"Invalid file suffix: {bedgraph_file}, read anyway...")

    # check file existence
    bedgraph_file = Path(bedgraph_file)
    if not bedgraph_file.exists():
        raise FileNotFoundError(bedgraph_file)

    # read bedgraph file
    lf = pl.scan_csv(
        bedgraph_file,
        separator="\t",
        has_header=False,
        comment_prefix="#",
        null_values=".",
    )

    # standard BEDGraph column names
    bedgraph_cols = list(columns.keys())

    # rename columns safely (only up to existing width)
    schema = lf.collect_schema()
    lf = lf.rename(
        {
            f"column_{i + 1}": bedgraph_cols[i]
            for i in range(min(len(bedgraph_cols), len(schema)))
        }
    )

    existing_cols = set(lf.collect_schema().names())

    # build cast expressions for existing columns
    cast_exprs = [
        pl.col(col).cast(dtype)
        for col, dtype in columns.items()
        if col in existing_cols
    ]

    lf = lf.with_columns(cast_exprs)

    return lf

def read_bed(
    bed_file: str,
    columns: dict[str, pl.DataType] = BED_COLS,
) -> pl.LazyFrame:
    """
    Read a BED or BED.GZ file using pysam and polars.

    This function returns a :class:`polars.LazyFrame`.

    The input file is assumed to be in BED format with **at least
    three columns**:

    - ``chrom`` (chromosome name)
    - ``start`` (0-based start position)
    - ``end`` (end position, exclusive)

    Any additional columns present in the file are preserved and
    automatically assigned standard BED column names when possible.

    Parameters
    ----------
    bed_file : str
        Path to the input BED file. Both uncompressed ``.bed`` and
        gzip-compressed ``.bed.gz`` files are supported.

    columns : dict[str, pl.DataType]
        Column names and data types to read.

    Returns
    -------
    polars.LazyFrame
        A lazily-evaluated Polars LazyFrame representing the BED file
        contents.

    Raises
    ------
    FileNotFoundError
        If the specified BED file does not exist.
    """
    import polars as pl
    from pathlib import Path
    import warnings

    # check suffix
    if not bed_file.endswith(".bed") and not bed_file.endswith(".bed.gz"):
        warnings.warn(f"Invalid file suffix: {bed_file}, read anyway...")

    # check file existence
    bed_file = Path(bed_file)
    if not bed_file.exists():
        raise FileNotFoundError(bed_file)

    # read bed file
    lf = pl.scan_csv(
        bed_file,
        separator="\t",
        has_header=False,
        comment_prefix="#",
        null_values=".",
    )

    # standard BED column names
    bed_cols = list(columns.keys())

    # rename columns safely (only up to existing width)
    schema = lf.collect_schema()
    lf = lf.rename(
        {
            f"column_{i + 1}": bed_cols[i]
            for i in range(min(len(bed_cols), len(schema)))
        }
    )

    existing_cols = set(lf.collect_schema().names())

    # build cast expressions for existing columns
    cast_exprs = [
        pl.col(col).cast(dtype)
        for col, dtype in columns.items()
        if col in existing_cols
    ]

    lf = lf.with_columns(cast_exprs)

    return lf

def read_indexed_bed(
    bed_file: str, 
    chrom: str, 
    start: int = 0, 
    end: int = 1e9,
    columns: dict[str, pl.DataType] = BED_COLS,
) -> pl.LazyFrame:
    """
    Read a indexed BED.GZ file using pysam and polars.

    This function returns a :class:`polars.LazyFrame` and does **not**
    read or parse the file immediately. The file will only be read
    when the lazy query is executed (e.g. via ``collect()``).

    The input file is assumed to be in BED format with **at least
    three columns**:

    - ``chrom`` (chromosome name)
    - ``start`` (0-based start position)
    - ``end`` (end position, exclusive)

    Any additional columns present in the file are preserved and
    automatically assigned standard BED column names when possible.

    Parameters
    ----------
    bed_file : str
        Path to the input BED.GZ file.

    chrom : str
        Chromosome (sequence name) to read.

    start : int
        Start coordinate of a region to read. Default is 0.

    end : int
        End coordinate of a region to read. Default is 1e9.

    columns : dict[str, pl.DataType]
        Column names and data types to read.

    Returns
    -------
    polars.LazyFrame
        A lazily-evaluated Polars LazyFrame representing the BED file
        contents.

    Raises
    ------
    FileNotFoundError
        If the specified BED file does not exist.
    """
    import polars as pl
    from pathlib import Path
    import warnings
    import pysam
    
    # check suffix
    if not bed_file.endswith(".bed.gz"):
        warnings.warn(f"Invalid file suffix: {bed_file}, read anyway...")

    # check file and index existence
    bed_file = Path(bed_file)
    if not bed_file.exists():
        raise FileNotFoundError(bed_file)
    index_file = Path(str(bed_file) + ".tbi")
    if not index_file.exists():
        raise FileNotFoundError(index_file)

    # open bed file
    bed = pysam.TabixFile(str(bed_file))

    # get number of columns
    first_line = next(bed.fetch(chrom, start, end))
    num_cols = len(first_line.strip().split("\t"))

    # build generator to convert tabix fetch output to tuple
    schema = list(columns.keys())[:num_cols]

    # build generator to convert tabix fetch output to tuple
    def fetch_to_rows():
        # first line already fetched, yield it
        yield tuple(first_line.strip().split("\t"))
        # remaining lines
        for line in bed.fetch(chrom, start, end):
            yield tuple(line.strip().split("\t"))

    # build lazy dataframe
    lf = pl.DataFrame(fetch_to_rows(), schema=schema).lazy()

    existing_cols = set(lf.collect_schema().names())

    # build cast expressions for existing columns
    cast_exprs = [
        pl.col(col).cast(dtype)
        for col, dtype in columns.items()
        if col in existing_cols
    ]

    lf = lf.with_columns(cast_exprs)

    # collect dataframe
    return lf


###
#
# read vampire annotation result (for single TR locus)
#
###
def read_anno(
    file: str,
    use_raw: bool = False,
    match_score: int = 2,
    mismatch_penalty: int = 4,
    gap_open_penalty: int = 7,
    gap_extend_penalty: int = 4,
) -> ad.AnnData:
    """
    Read a vampire annotation result file using pysam and polars.

    Parameters
    ----------
    file : str
        Path to the input annotation result file. e.g. example.annotation.tsv
    use_raw: bool
        Whether to read the real sequences instead of annotated motifs.
        This may dramatically increase the number of motifs. 
        Default is False.

    Returns
    -------
    anndata.AnnData
        An AnnData object containing the annotation result.
        Each row (obs) is a sample, each column (var) is a motif.
    
    Indexing:
        - obs index corresponds to samples (chrom)
        - var index corresponds to motifs
        - X[i, j] aligns with obs[i] and var[j]

    Structure
    ---------

    X:
        (n_obs × n_var) motif abundance / copy-number matrix
        X[i, j] = copy number of motif j in chromosome i

    obs:
        Sample metadata (n_obs × metadata)
        - length : int
        - copy_number : float
        - score : int

    var:
        Motif metadata (n_var × metadata)
        - motif : str
        - motif_length : int
        - copy_number : float
        - label : str

    varp:
        Motif-level pairwise relations (n_var × n_var)
        - motif_distance : int
        - rc_motif_distance : int

    uns:
        Unstructured genomic annotations (not aligned to X)
        - sequence : dict[str, str]
        - motif_array : dict[str, list[str]]
        - orientation_array : dict[str, list[str]]
    """
    import pyarrow
    import numpy as np
    import pandas as pd
    import anndata as ad
    from pathlib import Path
    from vampire._report_utils import(
        get_copy_number
    )

    # check file existence
    if not file.endswith(".annotation.tsv"):
        raise ValueError(f"Input file should be *.annotation.tsv, but found: {file}")
    anno_file = Path(file)
    if not anno_file.exists():
        raise FileNotFoundError(anno_file)
    concise_file = Path(file.replace(".annotation.tsv", ".concise.tsv"))
    if not concise_file.exists():
        raise FileNotFoundError(concise_file)
    motif_file = Path(file.replace(".annotation.tsv", ".motif.tsv"))
    if not motif_file.exists():
        raise FileNotFoundError(motif_file)
    dist_file = Path(file.replace(".annotation.tsv", ".distance.tsv"))
    if not dist_file.exists():
        raise FileNotFoundError(dist_file)
    if match_score < 0:
        raise ValueError("match_score must be positive!")
    if mismatch_penalty < 0:
        raise ValueError("mismatch_penalty must be positive!")
    if gap_open_penalty < 0:
        raise ValueError("gap_open_penalty must be positive!")
    if gap_extend_penalty < 0:
        raise ValueError("gap_extend_penalty must be positive!")

    # read
    anno_df: pl.DataFrame = pl.read_csv(anno_file, separator = "\t", has_header = True, null_values=".")
    concise_df: pl.DataFrame = pl.read_csv(concise_file, separator = "\t", has_header = True, null_values=".")
    motif_df: pl.DataFrame = pl.read_csv(motif_file, separator = "\t", has_header = True, null_values=".")
    dist_df: pl.DataFrame = pl.read_csv(dist_file, separator = "\t", has_header = True, null_values=".")

    # apply use_raw, this need to calculate motifs again
    if use_raw:
        anno_df, concise_df, motif_df, dist_df = make_raw(
            anno_df, concise_df, motif_df, dist_df,
            match_score = match_score,
            mismatch_penalty = mismatch_penalty,
            gap_open_penalty = gap_open_penalty,
            gap_extend_penalty = gap_extend_penalty,
        )
    
    # filter
    anno_df = anno_df.filter(pl.col("motif").is_not_null())

    chrom_order: list[str] = anno_df["chrom"].unique(maintain_order=True).to_list()
    order_df: pl.DataFrame = pl.DataFrame({
        "chrom": chrom_order,
        "order": range(len(chrom_order))
    })

    # make obs and meta information
    obs: pd.DataFrame = (
        concise_df
        .select([
            pl.col("chrom").alias("sample"),
            pl.col("length"),
            pl.col("copyNumber").alias("copy_number"),
            pl.col("score"),
        ])
        .to_pandas()
        .set_index("sample")
    )
    obs = obs.loc[chrom_order] # sort
    logger.debug("obs dataframe is created")

    # make var and meta information
    var: pd.DataFrame = (
        motif_df.select([
            pl.col("id").cast(pl.Utf8),
            pl.col("motif"),
            pl.col("motif").map_elements(lambda x: len(x), return_dtype=pl.Int64).alias("motif_length"),
            pl.col("copyNumber").alias("copy_number"),
            pl.col("label"),
        ])
        .to_pandas()
        .set_index("id")
    )
    var["label"] = var["label"].astype("category")
    motif_order: list[str] = var.index
    logger.debug("var dataframe is created")

    # make X matrix (represent the count of motif in each sample)
    motif2length: dict[str, int] = dict(zip(var.index, var["motif_length"]))
    anno_df: pl.DataFrame = anno_df.with_columns(
        pl.col("motif")
        .cast(pl.Utf8)
        .replace(motif2length)
        .cast(pl.Int64)
        .alias("motif_length")
    )

    # _use_raw stores the per-block copy number explicitly because the CIGAR
    # is simplified to a perfect match; avoid recomputing it from that CIGAR.
    if "copyNumber" in anno_df.columns:
        anno_df: pl.DataFrame = anno_df.with_columns(pl.col("copyNumber").alias("copy_number"))
    else:
        anno_df: pl.DataFrame = anno_df.with_columns(
            pl.struct(["motif_length", "cigar"])
            .map_elements(lambda x: get_copy_number(x["cigar"], x["motif_length"]), return_dtype=pl.Float64)
            .alias("copy_number")
        )
    X_df: pl.DataFrame = (
        anno_df
        .pivot(
            values="copy_number",
            index="chrom",
            on="motif",
            aggregate_function="sum"
        )
        .fill_null(0)
        .join(order_df, on="chrom", how="left")
        .sort("order")
        .drop("order")
    )
    # _use_raw may produce a motif catalog where a motif has no referencing
    # blocks (e.g., after filtering empty/ambiguous motifs). Ensure the X
    # matrix has exactly the columns declared in var.
    for col in motif_order:
        if col not in X_df.columns:
            X_df = X_df.with_columns(pl.lit(0.0).alias(col))
    X_df = X_df.select(["chrom"] + list(motif_order))
    X: np.ndarray = X_df.drop("chrom").to_numpy()
    del chrom_order, order_df
    logger.debug("X matrix is created")

    # make anndata object
    adata = ad.AnnData(X = X, obs = obs, var = var)
    logger.debug("anndata object is created")

    # make uns metadata - sequence
    seq_df: pl.DataFrame = (
        anno_df
        .group_by("chrom")
        .agg(
            pl.col("sequence").str.join("")
        )
    )
    seq_dict: dict[str, str] = dict(zip(seq_df["chrom"], seq_df["sequence"]))
    adata.uns["sequence"] = seq_dict
    del seq_dict
    logger.debug("added .uns['sequence']")

    # make uns metadata - motif_array
    motif_array_dict: dict[str, str] = dict(zip(concise_df["chrom"], concise_df["motif"])) # {"sample1" : "0,1,0,3,4"}
    motif_array_dict: dict[str, list[str]] = {c: motif_array_dict[c].split(",") for c in adata.obs.index} # {"sample1" : ["0", "1", "0", "3" ,"4"]}
    adata.uns["motif_array"] = motif_array_dict
    del motif_array_dict
    logger.debug("added .uns['motif_array']")

    # make uns metadata - orientation_array
    orientation_array_dict: dict[str, str] = dict(zip(concise_df["chrom"], concise_df["orientation"])) # {"sample1" : "+,+,-,-,+"}
    orientation_array_dict: dict[str, list[str]] = {c: orientation_array_dict[c].split(",") for c in adata.obs.index} # {"sample1" : ["+", "+", "-", "-" ,"+"]}
    adata.uns["orientation_array"] = orientation_array_dict
    del orientation_array_dict
    logger.debug("added .uns['orientation_array']")

    # make uns metadata - per-block copy numbers for accurate plotting of partial copies
    block_cn_df = (
        anno_df
        .group_by("chrom")
        .agg(pl.col("copy_number"))
    )
    block_cn_dict: dict[str, list[float]] = {
        chrom: [float(cn) if cn is not None else float("nan") for cn in cns]
        for chrom, cns in zip(block_cn_df["chrom"].to_list(), block_cn_df["copy_number"].to_list())
    }
    adata.uns["block_copy_number"] = block_cn_dict
    logger.debug("added .uns['block_copy_number']")

    # make varp (pairwise) - motif distance
    target_idx: np.ndarray = dist_df["target"].to_numpy()
    query_idx: np.ndarray = dist_df["query"].to_numpy()
    distance: np.ndarray = dist_df["distance"].to_numpy()
    is_rc: np.ndarray = dist_df["is_rc"].to_numpy()

    n: int = max(target_idx.max(), query_idx.max()) + 1
    mat_false: np.ndarray = np.zeros((n, n))
    mat_true: np.ndarray  = np.zeros((n, n))
    mat_false[target_idx[~is_rc], query_idx[~is_rc]] = distance[~is_rc]
    mat_false[query_idx[~is_rc], target_idx[~is_rc]] = distance[~is_rc]
    mat_true[target_idx[is_rc], query_idx[is_rc]] = distance[is_rc]
    mat_true[query_idx[is_rc], target_idx[is_rc]] = distance[is_rc]
    adata.varp["motif_distance"] = mat_false
    adata.varp["rc_motif_distance"] = mat_true
    logger.debug("added .varp['motif_distance'] and .varp['rc_motif_distance']")

    return adata