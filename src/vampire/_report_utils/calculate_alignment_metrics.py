from typing import List, Tuple
from collections import Counter

def calculate_alignment_metrics(ops: List[str], seq_len: int) -> Tuple[int, int]:
    """
    Calculate the alignment metrics
    Inputs:
        ops : List[str], atomic operations: '=', 'X', 'I', 'D', '/'
        seq_len : int, length of the sequence
    Outputs:
        match_count : int, match percentage
        indel_count : int, indel percentage
    """
    counts = Counter(ops)
    match_count = float(counts["="]) if "=" in counts else 0.0
    indel_count = 0.0
    if "I" in counts:
        indel_count += counts["I"]
    if "D" in counts:
        indel_count += counts["D"]
    return int(match_count / seq_len * 100), int(indel_count / seq_len * 100)