from ._make_stats import make_stats
from ._make_report import make_report
from ._stats_cigar import(
    ops_to_cigar,
    get_copy_number,
    calculate_alignment_metrics
)
from ._stats_seq import calculate_nucleotide_composition

__all__ = [
    make_stats,
    make_report,
    ops_to_cigar,
    get_copy_number,
    calculate_alignment_metrics,
    calculate_nucleotide_composition
]