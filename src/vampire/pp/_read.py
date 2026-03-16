import polars as pl
from pathlib import Path
import warnings
from typing import Dict

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

    This function returns a :class:`polars.DataFrame`.

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
) -> pl.DataFrame:
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