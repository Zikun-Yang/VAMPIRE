from typing import List, Tuple, Optional
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

def ops_to_cigar(ops: List[str], m: int) -> str:
    """
    Convert atomic ops to CIGAR string.
    Inputs:
        ops : List[str]
            atomic operations: '=', 'X', 'I', 'D', '/'
            - '=': match (seq and motif have same base)
            - 'X': mismatch (seq and motif have different bases)
            - 'I': insertion in seq (seq has extra base, motif has gap)
            - 'D': deletion in seq (seq has gap, motif has extra base)
            - '/': motif wrap separator (indicates end of one motif copy)
        m : int, length of the motif
    Outputs:
        cigar : str
            CIGAR string
    Rules:
        '=', 'X', 'I', 'D' are counted independently
        '/' is a standalone separator (motif wrap)
    """
    cigar: List[str] = []
    last_op: Optional[str] = None
    cur_pos: int = 0
    count: int = 0

    for op in ops:
        if last_op is None:
            last_op = op
            cur_pos += 1
            count = 1
        elif op == last_op:
            count += 1
            if op != 'I':
                cur_pos += 1
        else:
            cigar.append(f"{count}{last_op}")
            last_op = op
            count = 1
            if op != 'I':
                cur_pos += 1
        # motif wrap
        if cur_pos == m:
            cigar.append(f"{count}{last_op}/")
            cur_pos = 0
            last_op = None
            count = 0
    if last_op is not None:
        cigar.append(f"{count}{last_op}")

    return "".join(cigar)

def get_copy_number(cigar: str, m: int) -> float:
    """
    Get the copy number of the region
    Inputs:
        cigar : str, CIGAR string with '/' separators
        m : int, length of the motif
    Outputs:
        copy number : float, complete copies + fractional last copy
    """
    complete_num = cigar.count("/")
    last_copy_cigar = cigar.split("/")[-1]
    last_copy_length = 0.0
    num = ""
    for op in last_copy_cigar:
        if op.isdigit():
            num += op
        elif op in ["=", "X", "D"]:
            last_copy_length += int(num)
            num = ""
    cn = round(last_copy_length / m + complete_num, 1)
    return cn