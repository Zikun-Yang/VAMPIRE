import numpy as np
from typing import Tuple
import numba

global ENCODE_TABLE
ENCODE_TABLE = np.full(256, -1, dtype=np.int8)
ENCODE_TABLE[ord('A')] = 0
ENCODE_TABLE[ord('C')] = 1
ENCODE_TABLE[ord('G')] = 2
ENCODE_TABLE[ord('T')] = 3

def encode_seq_to_array(seq: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode a sequence into an integer array with homopolymer compression.
    A=0, C=1, G=2, T=3, N and other invalid characters are encoded as -1.
    Homopolymer runs (poly-A/T/C/G) are collapsed.
    Input:
        seq: str, the sequence to encode
    Output:
        Tuple[np.ndarray, np.ndarray]: (compressed_encoded_seq, counts)
        - compressed_encoded_seq: the encoded sequence with homopolymers collapsed
        - counts: array representing the number of bases in each homopolymer run
    Example:
        "ATAAACG" -> (array([0,3,0,1,2]), array([1,1,3,1,1]))
        which represents: A(1), T(1), A(3), C(1), G(1)
    """
    encoded_seq = ENCODE_TABLE[np.frombuffer(seq.encode(), dtype=np.uint8)]
    return encoded_seq

def decode_array_to_seq(encoded_seq: np.ndarray) -> str:
    """
    Decode an encoded sequence into a string
    Input:
        encoded_seq: np.ndarray, the encoded sequence, 0 - 3 for A, C, G, T, -1 is invalid
    Output:
        str, the decoded sequence
    """
    lut = np.array(['A', 'C', 'G', 'T'], dtype='<U1')
    return ''.join(lut[encoded_seq])

@numba.njit(cache=True)
def compress_homopolymers(encoded_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compress homopolymer runs in an encoded sequence and return compressed sequence with counts.
    Input:
        encoded_seq: np.ndarray, encoded sequence (A=0, C=1, G=2, T=3, invalid=-1)
    Output:
        np.ndarray, compressed sequence with homopolymers collapsed
        np.ndarray, counts for each base in compressed sequence
    """
    n = len(encoded_seq)
    if n == 0:
        return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.int32)
    
    # Pre-allocate arrays (worst case: no compression, same size)
    compressed = np.empty(n, dtype=np.int8)
    counts = np.empty(n, dtype=np.int32)
    
    compressed_idx = 0
    i = 0
    
    while i < n:
        current_base = encoded_seq[i]
        count = 1
        
        # Count consecutive identical bases (including invalid bases as separate)
        while i + count < n and encoded_seq[i + count] == current_base:
            count += 1
        
        # Store compressed base and count
        compressed[compressed_idx] = current_base
        counts[compressed_idx] = count
        compressed_idx += 1
        i += count
    
    # Trim to actual size
    return compressed[:compressed_idx], counts[:compressed_idx]

@numba.njit(cache=True)
def decompress_array(compressed_seq: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Decompress a compressed sequence and counts into an encoded sequence
    Input:
        compressed_seq: np.ndarray, compressed sequence
        counts: np.ndarray, counts
    Output:
        np.ndarray, the decompressed encoded sequence
    """
    return np.repeat(compressed_seq, counts)

@numba.njit(cache=True) # TODO CHANGE
def _encode_array_to_int_fast(encoded_seq: np.ndarray) -> np.int64:
    """
    Convert an encoded sequence (array) to an integer value for small k (k <= 31).
    Uses numba for acceleration. A=0, C=1, G=2, T=3, invalid=-1
    Input:
        encoded_seq: np.ndarray, encoded sequence (A=0, C=1, G=2, T=3, invalid=-1)
    Output:
        np.int64: the integer encoding of the encoded sequence, or -1 if invalid
    Note:
        Only works for k <= 31 (fits in int64). Returns -1 for invalid sequences.
    """
    k = len(encoded_seq)
    if k > 31:
        return -1  # cannot fit in int64
    value = np.int64(0)
    mask = (np.int64(1) << (2 * k)) - 1
    for i in range(k):
        b = encoded_seq[i]
        if b == -1 or b < 0 or b > 3:
            return -1
        value = ((value << 2) | np.int64(b)) & mask
    return value

def encode_array_to_int(encoded_seq: np.ndarray) -> int | None: # TODO CHANGE
    """
    Convert an encoded sequence (array) to an integer value, A=0, C=1, G=2, T=3, invalid=-1
    Input:
        encoded_seq: np.ndarray, encoded sequence (A=0, C=1, G=2, T=3, invalid=-1)
    Output:
        int | None: the integer encoding of the encoded sequence, or None if invalid
    Note:
        For k <= 31, uses numba-accelerated version. For k > 31, uses Python implementation.
    """
    k = len(encoded_seq)
    if k <= 31:
        result = _encode_array_to_int_fast(encoded_seq)
        return int(result) if result != -1 else None
    else:
        # for large k, use Python implementation
        value = 0
        mask = (1 << (2 * k)) - 1
        for i, b in enumerate(encoded_seq):
            if b == -1 or b < 0 or b > 3:
                return None
            value = ((value << 2) | int(b)) & mask
        return value

@numba.njit(cache=True)
def _decode_int_to_array_fast(value: np.int64, k: int) -> np.ndarray: # TODO CHANGE
    """
    Decode an integer value into an encoded sequence (array) for small k (k <= 31).
    Uses numba for acceleration. A=0, C=1, G=2, T=3, invalid=-1
    Input:
        value: np.int64, the integer to decode
        k: int, the length of the encoded sequence
    Output:
        np.ndarray, the decoded encoded sequence (array)
    Note:
        Only works for k <= 31 (fits in int64).
    """
    arr = np.empty(k, dtype=np.int8)
    val = value
    for i in range(k-1, -1, -1):
        arr[i] = val & 3
        val >>= 2
    return arr

def decode_int_to_array(value: int, k: int) -> np.ndarray: # TODO CHANGE
    """
    Decode an integer value into an encoded sequence (array), A=0, C=1, G=2, T=3, invalid=-1
    Input:
        value: int, the integer to decode
        k: int, the length of the encoded sequence
    Output:
        np.ndarray, the decoded encoded sequence (array)
    Note:
        For k <= 31, uses numba-accelerated version. For k > 31, uses Python implementation.
    """
    if k <= 31:
        return _decode_int_to_array_fast(np.int64(value), k)
    else:
        # for large k, use Python implementation
        arr = np.empty(k, dtype=np.int8)
        val = value
        for i in range(k-1, -1, -1):
            arr[i] = val & 3
            val >>= 2
        return arr

@numba.njit(cache=True)
def canonicalize_motif(encoded_motif):
    """
    Canonicalize motif using Booth algorithm (O(n)).
    Returns the lexicographically smallest cyclic rotation.
    Input:
        encoded_motif: np.ndarray, encoded motif (A=0, C=1, G=2, T=3)
    Output:
        np.ndarray: the canonicalized (lexicographically smallest) cyclic rotation
    """
    n = len(encoded_motif)
    if n == 0:
        return encoded_motif

    i, j, k = 0, 1, 0

    while i < n and j < n and k < n:
        a = encoded_motif[(i + k) % n]
        b = encoded_motif[(j + k) % n]

        if a == b:
            k += 1
        elif a > b:
            i = i + k + 1
            if i <= j:
                i = j + 1
            k = 0
        else:
            j = j + k + 1
            if j <= i:
                j = i + 1
            k = 0

    start = i if i < j else j

    res = np.empty(n, dtype=encoded_motif.dtype)
    for t in range(n):
        res[t] = encoded_motif[(start + t) % n]

    return res

@numba.njit(cache=True)
def is_periodic_seq(encoded_seq: np.ndarray) -> Tuple[bool, int]:
    """
    Check if a encoded sequence is periodic
    Input:
        encoded_seq: np.ndarray, encoded sequence (A=0, C=1, G=2, T=3)
    Output:
        bool: True if the encoded sequence is periodic
        int: the minimal period of the encoded sequence
    """
    k = len(encoded_seq)

    if k <= 1:
        return False, k

    # prefix function
    pi = np.zeros(k, dtype=np.int32)

    j = 0 # j is the length of the longest proper prefix of the substring
    for i in range(1, k): # i is the current position in the sequence
        while j > 0 and encoded_seq[i] != encoded_seq[j]:
            j = pi[j - 1]

        if encoded_seq[i] == encoded_seq[j]:
            j += 1

        pi[i] = j

    period = k - pi[k - 1]

    is_periodic = period < k and k % period == 0

    return is_periodic, period
