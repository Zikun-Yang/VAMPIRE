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