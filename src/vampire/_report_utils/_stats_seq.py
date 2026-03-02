from typing import Dict, Tuple
from collections import Counter
import math

def calculate_nucleotide_composition(seq: str) -> Tuple[Dict[str, int], float]:
    """
    Calculate nucleotide composition and Shannon entropy (base-2).
    Inputs:
        seq : str
    Outputs:
        nt_comp : Dict[str, int], nucleotide composition
        entropy : float, Shannon entropy (base-2)
    """
    n = len(seq)
    if n == 0:
        return {"A": 0, "C": 0, "G": 0, "T": 0}, 0.00
    
    counts = Counter(seq)
    entropy = 0.00
    for char in counts:
        p = counts[char] / float(n)
        if p > 0:
            entropy -= p * math.log2(p)
    raw = {
        "A": int(counts["A"] / n * 100),
        "C": int(counts["C"] / n * 100),
        "G": int(counts["G"] / n * 100),
        "T": int(counts["T"] / n * 100),
    }
    floored = {k: math.floor(v) for k, v in raw.items()}
    remain = 100 - sum(floored.values())

    # sort by fractional part from largest to smallest
    remainders = sorted(
        raw.items(),
        key=lambda x: x[1] - math.floor(x[1]),
        reverse=True
    )
    for k, _ in remainders[:remain]:
        floored[k] += 1

    nt_comp = floored

    return nt_comp, round(entropy, 2)