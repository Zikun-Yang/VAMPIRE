from ._read import(
    read_bed, 
    read_bedgraph,
    read_indexed_bed,
    read_anno,
)
from ._markdup import markdup

__all__ = [
    "read_bed", 
    "read_bedgraph",
    "read_indexed_bed",
    "read_anno",
    "markdup",
]