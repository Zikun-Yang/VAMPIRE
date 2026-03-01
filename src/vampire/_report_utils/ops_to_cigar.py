from typing import List, Optional

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