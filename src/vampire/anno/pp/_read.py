from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List, Dict
import polars as pl
import logging
import numba
import numpy as np

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
    columns: Dict[str, pl.DataType] = BEDGRAPH_COLS,
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

    columns : Dict[str, pl.DataType]
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
    columns: Dict[str, pl.DataType] = BED_COLS,
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

    columns : Dict[str, pl.DataType]
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
    columns: Dict[str, pl.DataType] = BED_COLS,
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

    columns : Dict[str, pl.DataType]
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
# read vampire scan result (for multiple TR loci)
#
###

###
#
# read vampire annotation result (for single TR locus)
#
###
def read_anno(
    file: str,
    use_raw: bool = False,
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
        - sequence : Dict[str, str]
        - motif_array : Dict[str, List[str]]
        - orientation_array : Dict[str, List[str]]
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

    # read
    anno_df: pl.DataFrame = pl.read_csv(anno_file, separator = "\t", has_header = True, null_values=".")
    concise_df: pl.DataFrame = pl.read_csv(concise_file, separator = "\t", has_header = True, null_values=".")
    motif_df: pl.DataFrame = pl.read_csv(motif_file, separator = "\t", has_header = True, null_values=".")
    dist_df: pl.DataFrame = pl.read_csv(dist_file, separator = "\t", has_header = True, null_values=".")

    # apply use_raw, this need to calculate motifs again
    if use_raw:
        anno_df, concise_df, motif_df, dist_df = _remake_results(anno_df, concise_df, motif_df, dist_df)

    chrom_order: List[str] = anno_df["chrom"].unique(maintain_order=True).to_list()
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
    motif_order: List[str] = var.index
    logger.debug("var dataframe is created")

    # make X matrix (represent the count of motif in each sample)
    motif2length: Dict[str, int] = dict(zip(var.index, var["motif_length"]))
    anno_df: pl.DataFrame = anno_df.with_columns(
        pl.col("motif")
        .cast(pl.Utf8)
        .replace(motif2length)
        .cast(pl.Int64)
        .alias("motif_length")
    )
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
            columns="motif",
            aggregate_function="sum"
        )
        .fill_null(0)
        .join(order_df, on="chrom", how="left")
        .sort("order")
        .drop("order")
        .select(["chrom"] + list(motif_order))
    )
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
            pl.col("sequence").str.concat("")
        )
    )
    seq_dict: Dict[str, str] = dict(zip(seq_df["chrom"], seq_df["sequence"]))
    adata.uns["sequence"] = seq_dict
    del seq_dict
    logger.debug("added .uns['sequence']")

    # make uns metadata - motif_array
    motif_array_dict: Dict[str, str] = dict(zip(concise_df["chrom"], concise_df["motif"])) # {"sample1" : "0,1,0,3,4"}
    motif_array_dict: Dict[str, List[str]] = {c: motif_array_dict[c].split(",") for c in adata.obs.index} # {"sample1" : ["0", "1", "0", "3" ,"4"]}
    adata.uns["motif_array"] = motif_array_dict
    del motif_array_dict
    logger.debug("added .uns['motif_array']")

    # make uns metadata - orientation_array
    orientation_array_dict: Dict[str, str] = dict(zip(concise_df["chrom"], concise_df["orientation"])) # {"sample1" : "+,+,-,-,+"}
    orientation_array_dict: Dict[str, List[str]] = {c: orientation_array_dict[c].split(",") for c in adata.obs.index} # {"sample1" : ["+", "+", "-", "-" ,"+"]}
    adata.uns["orientation_array"] = orientation_array_dict
    del orientation_array_dict
    logger.debug("added .uns['orientation_array']")

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

def _remake_results(
    anno_df: pl.DataFrame,
    concise_df: pl.DataFrame,
    motif_df: pl.DataFrame,
    dist_df: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    calculate results based on the raw sequences
    """
    from vampire._utils import encode_seq_to_array
    from vampire._report_utils import(
        get_copy_number
    )

    id2motif: Dict[int, str] = dict(zip(motif_df["id"], motif_df["motif"]))
    id2len: Dict[int, int] = {motif_id: len(motif) for motif_id, motif in id2motif.items()}

    # add copy number information
    anno_df = anno_df.with_columns(
        pl.col("motif").replace_strict(id2len, return_dtype = pl.Int64).alias("motif_length")
    )
    anno_df = anno_df.with_columns(
        pl.struct(["cigar", "motif_length"])
        .map_elements(lambda x: get_copy_number(x["cigar"], x["motif_length"]), return_dtype = pl.Float64)
        .alias("copyNumber")
    )

    anno_df = anno_df.with_columns(
        pl.struct(["motif", "sequence"])
        .map_elements(lambda x: _calculate_phase_difference(id2motif[x["motif"]], x["sequence"]), return_dtype = pl.Int64)
        .alias("phase_difference")
    ) # calculate phase difference
    anno_df = anno_df.with_columns(
        pl.struct(["sequence", "phase_difference"])
        .map_elements(lambda x: x["sequence"][x["phase_difference"]: ] + x["sequence"][: x["phase_difference"]], return_dtype = pl.Utf8)
        .alias("rolled_sequence")
    ) # get rolled sequence
    anno_df = anno_df.with_columns(
        pl.when(pl.col("cigar").str.contains("/"))
        .then(pl.col("rolled_sequence"))
        .otherwise(pl.col("motif").replace_strict(id2motif, return_dtype = pl.Utf8))
        .alias("motif")
    ) # get real motif

    # make *.motif.tsv
    motif_df = (
        anno_df
        .group_by("motif")
        .agg(pl.col("copyNumber").sum().round(1).alias("copyNumber"))
    )
    motif_df = motif_df.filter(pl.col("motif").is_not_null()).sort(["copyNumber"], descending=True).with_row_index("id")
    motif_df = (
        motif_df
        .with_columns(pl.lit("UNKNOWN").alias("label"))
        .select(["id", "motif", "copyNumber", "label"])
    )
    # make *.dist.tsv
    id2motif: Dict[int, np.ndarray] = {row["id"]: encode_seq_to_array(row["motif"]) for row in motif_df.iter_rows(named=True)}
    id2motif_rc: Dict[int, np.ndarray] = {row["id"]: encode_seq_to_array(_rc(row["motif"])) for row in motif_df.iter_rows(named=True)}
    motif_num: int = len(id2motif)
    rows: List[Dict] = [{
        "target": i,
        "query": j,
        "distance": _calculate_edit_distance_between_motifs(id2motif[i], id2motif[j]),
        "sum_copyNumber": motif_df["copyNumber"][i] + motif_df["copyNumber"][j],
        "is_rc": False
    } for i in range(motif_num) for j in range(i + 1, motif_num)]
    rows_rc: List[Dict] = [{
        "target": i,
        "query": j,
        "distance": _calculate_edit_distance_between_motifs(id2motif[i], id2motif_rc[j]),
        "sum_copyNumber": motif_df["copyNumber"][i] + motif_df["copyNumber"][j],
        "is_rc": True
    } for i in range(motif_num) for j in range(i, motif_num)]
    dist_df: pl.DataFrame = pl.DataFrame(rows + rows_rc)
    dist_df = dist_df.sort(["distance", "sum_copyNumber", "target", "query"]).select(["target", "query", "distance", "is_rc"])
    # make *.annotation.tsv
    anno_df = (
        anno_df
        .join(motif_df.select(["motif", "id"]), on="motif", how="left")
        .with_columns(pl.col("id").alias("motif"))
    )
    # make *.concise.tsv
    concise_df = (
        anno_df
        .group_by("chrom")
        .agg(
            pl.col("length").first().alias("length"),
            pl.col("start").min().alias("start"),
            pl.col("end").max().alias("end"),
            pl.col("motif")
                .drop_nulls()
                .str.join(",")
                .alias("motif"),
            pl.col("orientation")
                .drop_nulls()
                .str.join(",")
                .alias("orientation"),
            pl.col("score").sum().alias("score"),
            pl.col("cigar").str.join("").alias("cigar"),
            pl.col("motif").drop_nulls().last().alias("last_motif"),
            pl.col("copyNumber").sum().round(1).alias("copyNumber"),
        )
    )
    anno_df = anno_df.select(["chrom", "length", "start", "end", "motif", "orientation", "sequence", "score", "cigar"])
    concise_df = concise_df.select(["chrom", "length", "start", "end", "motif", "orientation", "copyNumber", "score", "cigar"])

    return anno_df, concise_df, motif_df, dist_df

def _rc(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N':'N'}
    return ''.join(complement[base] for base in reversed(seq))

MATCH_SCORE = 2
MISMATCH_PENALTY = 7
GAP_OPEN_PENALTY = 7
GAP_EXTEND_PENALTY = 7

@numba.njit(cache=True)
def _banded_dp_align(
    seq: np.ndarray,
    motif: np.ndarray,
    band_width: int,
    align_to_end: bool = False,
    anchor_row: int = -1,
    compare_row: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Inputs:
            seq: np.ndarray, sequence
            motif: np.ndarray, motif
            band_width: int, band width
            align_to_end: bool, whether to align to end
            anchor_row: int, 0-based anchor row; -1 if unused
            compare_row: int, 0-based compare row; -1 means no downstream / skip compare logic
        Outputs:
            score_array: np.ndarray, score array
            band_argmax_j: np.ndarray, band argmax j
            trace_M: np.ndarray, traceback matrix for match / mismatch
            trace_I: np.ndarray, traceback matrix for insertion
            trace_D: np.ndarray, traceback matrix for deletion
    """
    # score_array[i, s] = max M/I/D in band at DP row i+1; s in {0,1,2} -> M,I,D.
    # band_argmax_j[i, s] = motif column j (0-based) where that row-state max is attained
    # (first j in band scan order on ties, same as strict > updates).
    n = seq.shape[0]
    m = motif.shape[0]
    NEG_INF = -10 ** 9
    # Numba nopython cannot type i <= compare_row when compare_row may be None; use -1 sentinel.
    have_downstream = compare_row >= 0

    score_array = np.full((n, 3), NEG_INF, np.int32)
    band_argmax_j = np.full((n, 3), -1, np.int16)

    # ---- DP matrices ----
    M = np.full((n + 1, m), NEG_INF, np.int32)
    I = np.full((n + 1, m), NEG_INF, np.int32)
    D = np.full((n + 1, m), NEG_INF, np.int32)

    trace_M = np.full((n + 1, m), -1, np.int8)
    trace_I = np.full((n + 1, m), -1, np.int8)
    trace_D = np.full((n + 1, m), -1, np.int8)

    # ---- init ----
    for j in range(m):
        M[0, j] = 0

    # ---- precompute run-length of seq ----
    run_len = np.ones(n, dtype=np.int32)
    for i in range(1, n):
        if seq[i] == seq[i - 1]:
            run_len[i] = run_len[i - 1] + 1
        else:
            run_len[i] = 1

    # ---- parameters for scaling ----
    alpha = 0.5   # control the decay strength (can be adjusted)
    min_scale = 0.3  # lower bound, prevent gap too cheap

    best_score = NEG_INF

    for i in range(1, n + 1):

        j_center = (i - 1) % m
        j_start = max(0, j_center - band_width)
        j_end   = min(m, j_center + band_width + 1)

        si = seq[i - 1]
        cur_score_m = NEG_INF
        cur_score_i = NEG_INF
        cur_score_d = NEG_INF
        cur_j_m = -1
        cur_j_i = -1
        cur_j_d = -1

        # ---- current run-length ----
        rl = run_len[i - 1]

        # scale: 1 / (1 + alpha*(rl-1))
        scale = 1.0 / (1.0 + alpha * (rl - 1))
        if scale < min_scale:
            scale = min_scale

        gap_open_scaled   = int(GAP_OPEN_PENALTY * scale)
        gap_extend_scaled = int(GAP_EXTEND_PENALTY * scale)

        for j in range(j_start, j_end):

            prev_j = m - 1 if j == 0 else j - 1

            # ---- match/mismatch ----
            s = MATCH_SCORE if si == motif[j] else -MISMATCH_PENALTY

            # ---- M ----
            best_prev = M[i - 1, prev_j]
            state = 0

            v = I[i - 1, prev_j]
            if v > best_prev:
                best_prev = v
                state = 1

            v = D[i - 1, prev_j]
            if v > best_prev:
                best_prev = v
                state = 2

            M[i, j] = best_prev + s
            trace_M[i, j] = state

            # ---- I (gap in motif) ----
            open_i = M[i - 1, j] - gap_open_scaled
            ext_i  = I[i - 1, j] - gap_extend_scaled

            if open_i > ext_i:
                I[i, j] = open_i
                trace_I[i, j] = 0
            else:
                I[i, j] = ext_i
                trace_I[i, j] = 1

            # ---- D (gap in seq) ----
            open_d = M[i, prev_j] - gap_open_scaled
            ext_d  = D[i, prev_j] - gap_extend_scaled

            if open_d > ext_d:
                D[i, j] = open_d
                trace_D[i, j] = 0
            else:
                D[i, j] = ext_d
                trace_D[i, j] = 2

            # ---- record ----
            if M[i, j] > cur_score_m:
                cur_score_m = M[i, j]
                cur_j_m = j
            if I[i, j] > cur_score_i:
                cur_score_i = I[i, j]
                cur_j_i = j
            if D[i, j] > cur_score_d:
                cur_score_d = D[i, j]
                cur_j_d = j

        score_array[i - 1, 0] = cur_score_m
        band_argmax_j[i - 1, 0] = cur_j_m
        score_array[i - 1, 1] = cur_score_i
        band_argmax_j[i - 1, 1] = cur_j_i
        score_array[i - 1, 2] = cur_score_d
        band_argmax_j[i - 1, 2] = cur_j_d
        best_score = max(best_score, cur_score_m, cur_score_i, cur_score_d)

        # ---- early exit ----
        if not align_to_end:
            if have_downstream and i <= compare_row:
                continue
            # if have downstream, and the score of the compare row is higher than the score of the anchor row, break
            if have_downstream and score_array[anchor_row, :].max() <= score_array[compare_row, :].max():
                break
            # if already compute the compare row, and no likelihood to exceed the best score, break
            if (n - i) * MATCH_SCORE + max(cur_score_m, cur_score_i, cur_score_d) <= best_score:
                break
    
    return (
        score_array,
        band_argmax_j,
        trace_M, trace_I, trace_D,
    )

def _traceback_banded_roll_motif(
    trace_M: np.ndarray, trace_I: np.ndarray, trace_D: np.ndarray,
    best_i: int, best_j: int,
    m: int,
    seq: np.ndarray,
    motif: np.ndarray,
) -> Tuple[List[str], int, int]:
    """
    Inputs:
        trace_M: np.ndarray, traceback matrix for match / mismatch
        trace_I: np.ndarray, traceback matrix for insertion
        trace_D: np.ndarray, traceback matrix for deletion
        best_i: int, best index in seq
        best_j: int, best index in motif
        m: int, length of motif
        seq: np.ndarray, target sequence
        motif: np.ndarray, query motif
    Outputs:
        ops: List[str], atomic operations: '=', 'X', 'I', 'D', '/'
        start_i: int, starting position in seq (1-based DP index; 0 means start from beginning)
        start_j: int, starting position in motif (0-based index)
    Notes:
        state: 0=M (diagonal), 1=I (gap in motif), 2=D (gap in seq)
    """
    i, j = best_i, best_j
    state = 0

    ops: List[str] = []

    while i > 0: # index of seq, 1-based
        if state == 0:  # M
            prev_state = trace_M[i, j]
            ops.append("=" if seq[i - 1] == motif[j] else "X")
            i -= 1
            j = j - 1 if j > 0 else m - 1
            state = prev_state

        elif state == 1:  # I
            prev_state = trace_I[i, j]
            i -= 1
            state = prev_state
            ops.append("I")

        elif state == 2:  # D
            prev_state = trace_D[i, j]
            j = j - 1 if j > 0 else m - 1
            state = prev_state
            ops.append("D")

    ops.reverse()

    # i, 1-based DP index into seq; j, 0-based motif index before the first aligned motif base.
    start_i = i
    start_j = (j + 1) % m if m > 0 else 0

    return ops, start_i, start_j

def _calculate_edit_distance_between_motifs(m1: np.ndarray, m2: np.ndarray) -> int:
    """
    Calculate the edit distance between two motifs
    Input:
        motif_i: np.ndarray
        motif_j: np.ndarray
    Output:
        edit_distance: int
    """
    # m1 is the longer motif
    if len(m1) < len(m2):
        m1, m2 = m2, m1

    score_array, band_argmax_j, trace_M, trace_I, trace_D = _banded_dp_align(
        seq = m1,
        motif = m2,
        band_width = len(m2),
        align_to_end = True,
        anchor_row = -1, # no anchor
        compare_row = -1, # no compare
    )
                
    state: int= np.argmax(score_array[len(m1) - 1, :]) # 0 -> M, 1 -> I, 2 -> D
    best_j: int = band_argmax_j[len(m1) - 1, state]

    # calculate edit distance from traceback
    ops, _, _ = _traceback_banded_roll_motif(
        trace_M = trace_M,
        trace_I = trace_I,
        trace_D = trace_D,
        best_i = len(m1),
        best_j = best_j,
        m = len(m2),
        seq = m1,
        motif = m2
    )
    # count edit operations: 'X' (mismatch), 'I' (insertion), 'D' (deletion)
    edit_distance = sum(1 for op in ops if op in ['X', 'I', 'D'])
    return edit_distance

def _calculate_phase_difference(m1: str, m2: str):
    """
    calculate phase difference between motif1 and motif2, m1 = m2[phase_diff:] + m2[:phase_diff]
    Inputs:
        m1: str
        m2: str
    Outputs:
        phase_diff: int
    """
    from vampire._utils import encode_seq_to_array
    is_swap: bool = False
    if len(m1) < len(m2):
        is_swap = True
        m1, m2 = m2, m1

    encoded_m1: np.ndarray = encode_seq_to_array(m1)
    encoded_m2: np.ndarray = encode_seq_to_array(m2)

    score_array, band_argmax_j, _, _, _ = _banded_dp_align(
        seq=encoded_m1,
        motif=encoded_m2,
        band_width=len(m2),
        align_to_end=True,
        anchor_row=-1,
        compare_row=-1,
    )

    state = np.argmax(score_array[len(m1) - 1, :])
    best_j = band_argmax_j[len(m1) - 1, state]
    if is_swap:
        phase_diff = len(m2) - (best_j + 1)
    else:
        phase_diff = best_j + 1

    return phase_diff