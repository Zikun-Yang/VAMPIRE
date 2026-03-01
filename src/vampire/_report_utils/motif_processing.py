def collapse_long_motif(motif: str, max_length: int = 10, sep: str = "..") -> str:
    """
    Collapse long motif into shorter motifs
    Input:
        motif: str
        max_length: int, the maximum length of the collapsed motif
        sep: str, the separator between the two parts
    Output:
        motif: str
    """
    motif_length = len(motif)
    if motif_length <= max_length:
        return motif
    
    # split the motif into two parts
    l_length = (max_length - len(sep)) // 2
    r_length = max_length - len(sep) - l_length
    l_motif = motif[:l_length]
    r_motif = motif[-r_length:]
    
    return l_motif + sep + r_motif

def canonicalize_motif(motif: str) -> str:
    """
    Canonicalize motif using Booth algorithm (O(n)).
    Returns the lexicographically smallest cyclic rotation.
    """
    if not motif:
        return motif

    s = motif * 2
    n = len(motif)

    i, j, k = 0, 1, 0
    while i < n and j < n and k < n:
        if s[i + k] == s[j + k]:
            k += 1
        elif s[i + k] > s[j + k]:
            i = i + k + 1
            if i <= j:
                i = j + 1
            k = 0
        else:
            j = j + k + 1
            if j <= i:
                j = i + 1
            k = 0
    start = min(i, j)
    return s[start:start + n]