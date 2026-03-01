import polars as pl
from pathlib import Path

def read_bedgraph(bedgraph_file: str) -> pl.LazyFrame:
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
        raise Warning(f"Invalid file suffix: {bedgraph_file}, read anyway...")

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
    bedgraph_cols = [
        "chrom",
        "start",
        "end",
        "value",
    ]

    # rename columns safely (only up to existing width)
    schema = lf.collect_schema()
    lf = lf.rename(
        {
            f"column_{i + 1}": bedgraph_cols[i]
            for i in range(min(len(bedgraph_cols), len(schema)))
        }
    )

    return lf

def read_bed(bed_file: str) -> pl.LazyFrame:
    """
    Lazily read a BED or BED.GZ file using Polars.

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
        raise Warning(f"Invalid file suffix: {bed_file}, read anyway...")

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

    # standard BED column names (BED3+)
    bed_cols = [
        "chrom",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "thickStart",
        "thickEnd",
        "itemRgb",
        "blockCount",
        "blockSizes",
        "blockStarts",
    ]

    # rename columns safely (only up to existing width)
    schema = lf.collect_schema()
    lf = lf.rename(
        {
            f"column_{i + 1}": bed_cols[i]
            for i in range(min(len(bed_cols), len(schema)))
        }
    )

    return lf