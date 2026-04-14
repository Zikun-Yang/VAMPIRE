#! /usr/bin/env python3

# data processing
from math import sqrt
import numpy as np
from Bio import SeqRecord, SeqIO
from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view  # rolling median
# multiprocessing
from tqdm import tqdm
from multiprocessing import Pool
# type hints
from typing import List, Dict, Tuple, Any, Optional
# basic packages for file operations, logging
import ast
import csv
import time
import logging
import heapq
import shutil
from pathlib import Path
# numba packages for speed optimization
import numba
from numba.typed import Dict

from vampire._report_utils import(
    ops_to_cigar,
    get_copy_number,
    calculate_alignment_metrics,
    calculate_nucleotide_composition
)

# type definitions, the coordinates are 0-based and closed interval
Region = List[Any] # [win_id, chrom, start, end, ksizes, score, periods] : [int, str, int, int, List[int], float, List[Tuple(int, int)]]
RegionWithMotifAndSeq = List[Any] # [win_id, chrom, start, end, ksizes, score, periods, seq, motifs] : [int, str, int, int, List[int], float, List[Tuple(int, int)], str, List[str]]

logger = logging.getLogger(__name__)

"""
#
# codes for reading and splitting fasta file
#
"""
def read_fasta(fasta_file: str) -> SeqIO.FastaIO:
    """
    Read fasta file using BioPython
    Input:
        fasta_file: str
    Output:
        fasta: SeqIO.FastaIO
    """
    fasta = SeqIO.parse(fasta_file, "fasta")
    return fasta

def split_fasta_by_window(fasta: list[SeqRecord], seq_win_size: int, seq_ovlp_size: int, job_dir: str) -> List[str]:
    """
    Split fasta file into windows of the given size and overlap
    Input:
        fasta: list[SeqRecord]
        seq_win_size: int
        seq_ovlp_size: int
        job_dir: str
    Output:
        window_filepaths: List[str]
        for each window, the format is [window_id, chrom, start, end, seq], seq is upper case DNA string
    """
    if seq_ovlp_size >= seq_win_size:
        raise ValueError("Overlap size must be less than window size")

    window_filepaths: List[str] = []
    window_num = 0
    for record in fasta:
        seq_len = len(record.seq)
        if logger.isEnabledFor(logging.DEBUG):
            windows = ",".join(f"{i}-{min(i + seq_win_size, seq_len)}" for i in range(0, seq_len, seq_win_size - seq_ovlp_size))
            logger.debug(f"Splitting record: {record.id}, windows: {windows}")
        seq = str(record.seq).upper()
        for i in range(0, seq_len, seq_win_size - seq_ovlp_size):
            window = seq[i : i + seq_win_size]
            if i != 0 and len(window) < seq_ovlp_size:
                break

            output_filepath = f"{job_dir}/windows/window_{window_num + 1}.tsv"
            with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t', lineterminator='\n')
                writer.writerow([window_num + 1, record.id, i, min(i+seq_win_size, seq_len), window])
            window_filepaths.append(output_filepath)
            window_num += 1
            
    return window_filepaths

"""
#
# codes for sequence processing utilities
#
"""
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
def encode_array_to_int_fast(encoded_seq: np.ndarray) -> np.int64:
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
        result = encode_array_to_int_fast(encoded_seq)
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
def decode_int_to_array_fast(value: np.int64, k: int) -> np.ndarray: # TODO CHANGE
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
        return decode_int_to_array_fast(np.int64(value), k)
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

def get_most_frequent_kmer(encoded_seq: np.ndarray, k: int) -> Tuple[np.ndarray | None, int]:
    """
    Get the most frequent canonical k-mer in the sequence
    Input:
        encoded_seq: np.ndarray, the encoded sequence to find the most frequent canonical k-mer
        k: int, the length of the k-mer
    Output:
        mf_array: np.ndarray | None, the most frequent canonical k-mer
        mf_count: int, the count of the most frequent canonical k-mer
    Note:
        Uses optimized strategies based on k value:
        - For k <= 31: uses integer encoding with numba acceleration
        - For k > 31: uses tuple-based encoding to avoid large integer operations
    """
    # input validation
    n = len(encoded_seq)
    if k <= 0 or n < k:
        return None, 0

    # choose strategy based on k value
    USE_INT_ENCODING = (k <= 31)  # integer encoding only works for k <= 31 (fits in int64)
    
    # cache size limits to prevent memory explosion
    MAX_CACHE_SIZE = 10000  # maximum number of entries in each cache
    
    if USE_INT_ENCODING:
        counts = defaultdict(int)
        mf_canonical_kmer_int: int | None = None
        mf_count: int = 0
        low_complex_cache: dict[int, int] = {}  # cache period values
        kmer_to_canonical_cache: dict[int, int] = {}  # cache kmer_int -> canonical_int mapping
        canonical_array_cache: dict[int, np.ndarray] = {}  # cache canonical array for periodic check
        mask = (1 << (2 * k)) - 1  # mask to keep the last k bases (2 bits per base)
        kmer_int: int = 0
        valid_bases: int = 0
        
        # calculate maximum possible k-mer count for early exit
        MAX_POSSIBLE_COUNT = len(encoded_seq) - k + 1
        EARLY_EXIT_THRESHOLD = MAX_POSSIBLE_COUNT * 0.5

        for b in encoded_seq:
            # check for invalid bases: -1 or values outside [0, 3]
            if b == -1 or b < 0 or b > 3:
                valid_bases = 0
                kmer_int = 0
                continue
            
            # update k-mer using bit operations
            kmer_int = ((kmer_int << 2) | int(b)) & mask
            valid_bases += 1
            if valid_bases < k:  # skip if the k-mer contains invalid bases
                continue
            
            # get canonical k-mer int (with caching)
            if kmer_int not in kmer_to_canonical_cache:
                # limit cache size to prevent memory explosion
                if len(kmer_to_canonical_cache) >= MAX_CACHE_SIZE:
                    kmer_to_canonical_cache.clear()
                    canonical_array_cache.clear()
                    low_complex_cache.clear()  # keep in sync with canonical caches
                
                # decode, canonicalize, and encode in one step
                kmer_array = decode_int_to_array(kmer_int, k)
                kmer_canonical_array = canonicalize_motif(kmer_array)
                encoded = encode_array_to_int(kmer_canonical_array)
                # cache
                kmer_to_canonical_cache[kmer_int] = encoded
                canonical_array_cache[encoded] = kmer_canonical_array  # cache array for periodic check
            kmer_canonical_int = kmer_to_canonical_cache[kmer_int]
            if kmer_canonical_int is None:
                continue
            
            # count
            cnt = counts[kmer_canonical_int] + 1
            counts[kmer_canonical_int] = cnt

            if cnt > mf_count:
                # check if periodic (with caching)
                if kmer_canonical_int not in low_complex_cache:
                    # limit cache size to prevent memory explosion
                    if len(low_complex_cache) >= MAX_CACHE_SIZE:
                        low_complex_cache.clear()
                    # get canonical array from cache
                    kmer_canonical_array = canonical_array_cache.get(kmer_canonical_int)
                    is_periodic, period = is_periodic_seq(kmer_canonical_array)
                    low_complex_cache[kmer_canonical_int] = period
                else:
                    period = low_complex_cache[kmer_canonical_int]
                    is_periodic = period < k and k % period == 0
                
                if is_periodic:
                    continue
                mf_canonical_kmer_int = kmer_canonical_int
                mf_count = cnt
                
                # early exit: if count exceeds threshold, no k-mer can be more frequent
                if mf_count >= EARLY_EXIT_THRESHOLD:
                    break

        # fallback: if all k-mers are periodic, return the most frequent k-mer # TODO: could be improved
        if mf_canonical_kmer_int is None and counts:
            mf_canonical_kmer_int = max(counts.items(), key=lambda x: x[1])[0]
            mf_count = counts[mf_canonical_kmer_int]

        # decode back to sequence
        mf_array = decode_int_to_array(mf_canonical_kmer_int, k)
        
    else:
        counts = defaultdict(int)
        mf_canonical_kmer_tuple: Tuple[int, ...] | None = None
        mf_canonical_array: np.ndarray | None = None  # cache array to avoid final conversion
        mf_count: int = 0
        low_complex_cache: dict[Tuple[int, ...], int] = {}  # cache period values
        canonical_cache: dict[bytes, Tuple[Tuple[int, ...], np.ndarray]] = {}  # bytes -> (tuple, array)
        
        # use an efficient sampling way to find the most frequent k-mer
        MAX_POSSIBLE_COUNT = len(encoded_seq) - k + 1
        SAMPLING_SIZE = int(sqrt(MAX_POSSIBLE_COUNT))
        EARLY_EXIT_THRESHOLD = SAMPLING_SIZE * 0.5
        SAMPLING_STEP = int(MAX_POSSIBLE_COUNT / SAMPLING_SIZE)
        
        # pre-calculate valid count
        valid = 0
        for j in range(k - 1):
            if 0 <= encoded_seq[j] <= 3:
                valid += 1
            else:
                valid = 0
        
        # sliding window to extract k-mers, kmer is [i-k+1, i]
        for i in range(k - 1, len(encoded_seq)):
            # update valid count
            if 0 <= encoded_seq[i] <= 3:
                valid += 1
            else:
                valid = 0

            # skip if the k-mer contains invalid bases
            if valid < k:
                continue

            # sampling
            if i % SAMPLING_STEP != 0:
                continue
            
            # use view to extract k-mer (avoid copy)
            kmer_view = encoded_seq[i - k + 1:i + 1]
            kmer_bytes = kmer_view.tobytes()
            # check cache for canonical form
            if kmer_bytes not in canonical_cache:
                # limit cache size to prevent memory explosion
                if len(canonical_cache) >= MAX_CACHE_SIZE:
                    canonical_cache.clear()
                    low_complex_cache.clear()  # keep in sync
                # only copy when needed for canonicalize (canonicalize doesn't modify input)
                kmer_array = kmer_view.copy()
                canonical_array = canonicalize_motif(kmer_array)
                # cache both tuple (for dict key) and array (for periodic check)
                canonical_cache[kmer_bytes] = (tuple(canonical_array), canonical_array)
            kmer_canonical_tuple, kmer_canonical_array = canonical_cache[kmer_bytes]
            
            # count (use += for slightly better performance)
            counts[kmer_canonical_tuple] += 1
            cnt = counts[kmer_canonical_tuple]

            if cnt > mf_count:
                # check if periodic (with caching)
                if kmer_canonical_tuple not in low_complex_cache:
                    # limit cache size to prevent memory explosion
                    if len(low_complex_cache) >= MAX_CACHE_SIZE:
                        low_complex_cache.clear()
                    # fast path: check homopolymer (all bases same) before calling is_periodic_seq
                    # use vectorized operation for better performance (O(k) but faster constant factor)
                    is_homopolymer = np.all(kmer_canonical_array == kmer_canonical_array[0])
                    
                    if is_homopolymer:
                        # homopolymer is periodic with period 1
                        period = 1
                        is_periodic = True
                    else:
                        # use cached array directly (avoid np.array conversion)
                        is_periodic, period = is_periodic_seq(kmer_canonical_array)
                    low_complex_cache[kmer_canonical_tuple] = period
                else:
                    period = low_complex_cache[kmer_canonical_tuple]
                    is_periodic = period < k and k % period == 0
                
                if is_periodic:
                    continue
                mf_canonical_kmer_tuple = kmer_canonical_tuple
                mf_canonical_array = kmer_canonical_array  # cache array to avoid final conversion
                mf_count = cnt
                
                # early exit: if count exceeds threshold, no k-mer can be more frequent
                if mf_count >= EARLY_EXIT_THRESHOLD:
                    break

        # fallback: if all k-mers are periodic, return the most frequent k-mer # TODO: could be improved
        if mf_canonical_kmer_tuple is None:
            ###print(f"counts: {counts}")
            if not counts:
                ###print(f"encoded_seq {encoded_seq}, k {k}")
                pass
            mf_canonical_kmer_tuple = max(counts.items(), key=lambda x: x[1])[0]
            mf_count = counts[mf_canonical_kmer_tuple]
            mf_canonical_array = np.array(mf_canonical_kmer_tuple, dtype=np.int8)
    
        mf_array = mf_canonical_array

    return mf_array, mf_count

"""
#
# codes for calling repeats
#
"""
# main entry
def call_regions(task: Tuple[str, str, List[int], int, int, int]) -> str:
    """
    Call the nonoverlapped tandem repeat regions for the given sequence
    Input:
        task: Tuple[str, str, List[int], int, int, int, int], includes:
            job_dir: str, job directory path
            window_filepath: str, window file path
            ksizes: list[int], ksize list
            max_dist: int, maximum distance to call regions
            score_vision_size: int, window length to compute smoothness score
            min_smoothness: int, minimum smoothness score to call regions
    Output:
        result: str, raw region file path, the format is:
            List[window_id, chrom, start, end, k, score, period], start and end are 1-based and include the borders
    """
    job_dir, window_filepath, ksizes, max_dist, score_vision_size, min_smoothness = task
    # read window information
    csv.field_size_limit(1024 * 1024 * 1024)  # set as 1 GB to avoid overflow
    with open(window_filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            win_id, chrom, win_start, win_end, seq = row
    win_start, win_end = int(win_start), int(win_end)
    encoded_seq = encode_seq_to_array(seq)
    win = (win_id, chrom, win_start, win_end, encoded_seq)

    # call raw regions
    raw_rgns: List[Region] = call_raw_rgns(job_dir, win, ksizes, max_dist, score_vision_size, min_smoothness)
    logger.debug(f"window {win_id}: call_raw_rgns finished")
    
    # if no raw regions found, return None
    if not raw_rgns:
        return None
    
    # sort by coordinates (start from small to large, end from large to small)
    raw_rgns.sort(key=lambda x: (x[2], -x[3]))

    # merge overlapped or contained raw regions, merge k into list, use max score
    merged_rgns: List[Region] = merge_rgns(raw_rgns, include_offset=True)
    logger.debug(f"window {win_id}: merge_rgns finished")

    # return None if no merged regions
    if not merged_rgns:
        return None

    # remove concatemer motifs
    merged_rgns = remove_concatemer(merged_rgns, encoded_seq)
    logger.debug(f"window {win_id}: remove_concatemer finished")
    """
    for rgn in merged_rgns:
        print(f"raw merged region: {rgn}")
    #"""

    # write merged raw regions to file
    output_filepath = f"{job_dir}/raw_rgns/window_{win_id}.tsv"
    with open(output_filepath, 'w', newline='', encoding='utf-8') as fi:
        writer = csv.writer(fi, delimiter='\t', lineterminator='\n')
        """
        tmp = [[rgn[0], 
                rgn[1], 
                encoded_to_raw_start_coord[rgn[2]], 
                encoded_to_raw_end_coord[rgn[3]], 
                rgn[4], 
                rgn[5], 
                rgn[6]] for rgn in merged_rgns]
        """
        writer.writerows(merged_rgns)
    logger.debug(f"window {win_id}: write_raw_rgns finished")
    ### logger.info(f"debug ! merged_rgns: {merged_rgns}") # TODO

    # polish region borders
    polished_rgns: List[Region] = polish_rgns(merged_rgns, win)
    logger.debug(f"window {win_id}: polish_rgns finished")

    # sort by coordinates
    polished_rgns.sort(key=lambda x: (x[2], -x[3]))

    # merge overlapped polished regions
    merged_rgns: List[Region] = merge_rgns(polished_rgns, include_offset=False)
    logger.debug(f"window {win_id}: merge_rgns finished")

    # filter polished regions that are too short
    final_rgns: List[Region] = list(filter(lambda x: x[3] - x[2] + 1 >= MIN_LEN, merged_rgns))
    logger.debug(f"window {win_id}: filter_rgns finished, total {len(final_rgns)} regions left")
    
    # remove concatemer motifs
    final_rgns = remove_concatemer(final_rgns, encoded_seq)
    logger.debug(f"window {win_id}: polished regions remove_concatemer finished")

    # get motif and sequence, transform 0-based coordinates (closed interval) to 1-based global coordinates (closed interval)
    rgns_with_motif_and_seq: List[RegionWithMotifAndSeq] = get_motif_and_seq(final_rgns, win)
    logger.debug(f"window {win_id}: get_motif_and_seq finished")

    # write polished regions to file
    output_filepath = f"{job_dir}/polished_rgns/window_{win_id}.tsv"
    with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(rgns_with_motif_and_seq)

    logger.debug(f"window {win_id}: call_region finished")

    return output_filepath

def get_mode(nums: List[int]) -> int | None:
    """
    Get mode from a list of elements using numpy
    Input:
        nums: List[int]
    Output:
        mode: int|None, return None if no mode is found
    """
    if isinstance(nums, np.ndarray):
        nums = nums[~np.isnan(nums)]
    else: # List[int]
        nums = [num for num in nums if num is not None]
    
    match len(nums):
        case 0:
            return None
        case 1:
            return int(nums[0])
        case _:
            nums = np.asarray(nums, dtype=np.int64)
            vals, counts = np.unique(nums, return_counts=True)
            max_freq = counts.max()
            return int(vals[counts == max_freq][0])

def get_most_frequent_period_tuple(period_tuples: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Get the most frequent period tuple from a list of period tuples based on the length
    Input:
        period_tuples: List[Tuple[int, int]]
    Output:
        Tuple[int, int]: the most frequent period tuple
    """
    mf_p, mf_l = None, 0
    for p, l in period_tuples:
        if l > mf_l:
            mf_p = p
            mf_l = l
    return mf_p, mf_l

def merge_period_tuples(period_tuples1: List[Tuple[int, int]], period_tuples2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge two lists of period tuples
    Input:
        period_tuples1: List[Tuple[int, int]]
        period_tuples2: List[Tuple[int, int]]
    Output:
        List[Tuple[int, int]]: the merged period tuples, sorted by period
    """
    period_dict = {}
    for period, length in period_tuples1:
        if period in period_dict:
            period_dict[period] += length
        else:
            period_dict[period] = length
    for period, length in period_tuples2:
        if period in period_dict:
            period_dict[period] += length
        else:
            period_dict[period] = length
    return sorted([(p, l) for p, l in period_dict.items()])

@numba.njit(cache=True)
def _get_largest_confident_period_by_k(k: int, alpha: float = 0.1) -> int:
    """
    Get the largest confident period by k-mer
    Input:
        k: int
        alpha: float, confidence level, default is 0.1
    Output:
        int: largest confident period, the formula is 4 ** k * alpha
    """
    threshold = int(4 ** k * alpha)
    return threshold

@numba.njit(cache=True)
def calculate_distance(encoded_seq: np.ndarray, ksize: int = 17, max_dist: int = 10000) -> np.ndarray:
    """
    Calculate the distance of the given sequence
    Input:
        encoded_seq: np.ndarray, the encoded sequence, use encode_seq() to encode the sequence, A=0, C=1, G=2, T=3, N=-1
        ksize: int
        max_dist: int, maximum distance to record, use np.nan if exceeding
    Output:
        np.ndarray: the distance array, dtype=np.float32 with np.nan for missing values
    """
    n = len(encoded_seq)
    n_positions = n - ksize + 1
    if n_positions <= 0:
        return np.full(0, np.nan, dtype=np.float32)
    
    distances = np.full(n_positions, np.nan, dtype=np.float32)
    table_size = 1 << 20 if ksize > 10 else 1 << (2 * ksize)
    mask = table_size - 1
    
    prev_pos_table = np.full(table_size, -1, dtype=np.int32)
    # tag_table stores the complete 64-bit encoding, used to resolve conflicts (only encoding consistent is the same k-mer)
    tag_table = np.full(table_size, -1, dtype=np.int64)

    current_encoding = np.int64(0)
    bit_mask = (np.int64(1) << (2 * ksize)) - 1

    # preheat the first window
    n_count = 0
    for i in range(ksize):
        val = encoded_seq[i]
        if val == -1:
            n_count += 1
        current_encoding = (current_encoding << 2) | max(val, 0) # if current base is N, add 00 to the encoding

    # sliding window
    for i in range(n_positions):
        if n_count == 0:
            h = (current_encoding ^ (current_encoding >> 16)) & mask

            if tag_table[h] == current_encoding:
                prev_pos = prev_pos_table[h]
                dist = i - prev_pos
                if dist < max_dist:
                    distances[i] = dist

            tag_table[h] = current_encoding
            prev_pos_table[h] = i
        else:
            distances[i] = np.nan
        # rolling update
        if i + ksize < n:
            old_val = encoded_seq[i]
            new_val = encoded_seq[i + ksize]
            # update n_count
            if old_val == -1:
                n_count -= 1
            if new_val == -1:
                n_count += 1
            # update encoding
            current_encoding = ((current_encoding << 2) | max(new_val, 0)) & bit_mask

    return distances

@numba.njit(cache=True)
def split_regions_by_rolling_median(
    starts: np.ndarray, ends: np.ndarray, 
    rolling_median: np.ndarray, 
    max_output_size: int,
    relative_threshold: float = 0.3,
    min_consecutive_changes: int = 2
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Split regions by rolling_median value changes (numba accelerated)
    Uses relative threshold and requires consecutive changes to avoid over-splitting
    Inputs:
        starts, ends: original region start and end indices
        rolling_median: rolling_median array
        max_output_size: maximum size for output arrays (should be >= len(starts) * 2)
        relative_threshold: float, relative change threshold (default 0.05 = 5%)
        min_consecutive_changes: int, minimum consecutive positions with changes required to split (default 2)
    Outputs:
        new_starts, new_ends: split regions (arrays of size max_output_size, but only first count elements are valid)
        count: number of valid regions in output arrays
    """
    new_starts = np.zeros(max_output_size, dtype=np.int64)
    new_ends = np.zeros(max_output_size, dtype=np.int64)
    count = 0
    
    n_regions = len(starts)
    n_median = len(rolling_median)
    
    for i in range(n_regions):
        # check if we've reached max_output_size
        if count >= max_output_size:
            break
        
        start_idx = starts[i]
        end_idx = ends[i]
        
        # boundary check
        if start_idx >= n_median or end_idx < 0 or start_idx > end_idx:
            continue
        
        region_len = end_idx - start_idx + 1
        
        if region_len == 1:
            # single element, keep as is
            if count < max_output_size:
                new_starts[count] = start_idx
                new_ends[count] = end_idx
                count += 1
        else:
            # split region at change points with relative threshold and consecutive changes
            current_start = start_idx
            j = start_idx
            
            while j <= end_idx:
                # check if we've reached max_output_size
                if count >= max_output_size:
                    break
                
                # check if there's a significant change at position j
                median_j = rolling_median[j]
                median_next = rolling_median[j + 1]
                
                # relative change; avoid div by zero when median_j == 0
                denom = abs(median_j)
                if denom < 1e-12:
                    denom = 1e-12
                relative_change = abs(median_j - median_next) / denom
                
                if relative_change > relative_threshold:
                    # check if we have consecutive changes starting from position j
                    consecutive_count = 1
                    check_pos = j + 2  # start checking from j+2
                    
                    # count consecutive positions with significant changes relative to initial value
                    while check_pos <= end_idx and consecutive_count < min_consecutive_changes:
                        median_curr = rolling_median[check_pos]
                        relative_change_from_base = abs(median_curr - median_j) / denom
                        
                        if relative_change_from_base > relative_threshold:
                            consecutive_count += 1
                            check_pos += 1
                        else:
                            break
                    
                    # only split if we have enough consecutive changes
                    if consecutive_count >= min_consecutive_changes:
                        # end of current sub-region at j
                        if count < max_output_size:
                            new_starts[count] = current_start
                            new_ends[count] = j
                            count += 1
                        current_start = j + 1
                # continue checking from next position
                j += 1
            
            # add the last sub-region (from current_start to end_idx)
            if count < max_output_size:
                new_starts[count] = current_start
                new_ends[count] = end_idx
                count += 1
    
    return new_starts, new_ends, count

@numba.njit(cache=True)
def is_possible_concatemer(period1: int, period2: int, threshold: float = 0.2) -> bool:
    """
    Check if period1 is a possible concatemer of period2
    Input:
        period1: int, longer period
        period2: int, shorter period
        threshold: float, default is 0.2
    Output:
        is_possible_concatemer: bool
    """
    period1, period2 = float(period1), float(period2)
    if period1 / period2 > 1.5:
        t = period1 / period2 % 1
        if t <= threshold or t >= 1 - threshold:
            return True
    return False

# Step 1: call raw regions
def call_on_compressed(
    job_dir: str, 
    win: Tuple[int, str, int, int, np.ndarray],
    ksize: int, 
    max_dist: int, 
    score_vision_size: int, 
    min_smoothness: int
    ) -> List[Region]:
    """
    Call the raw regions for the given sequence
    Input:
        job_dir: str, job directory path
        win: Tuple[int, str, int, int, np.ndarray]
        ksize: int
        max_dist: int, maximum distance to call regions
        score_vision_size: int, window length to compute smoothness score
        min_smoothness: int, minimum smoothness score to call regions
    Output:
        raw_rgns: List[Region]
    """
    raw_rgns: List[Region] = []
    piece_rgns: List[Region] = []
    win_id, chrom, win_start, win_end, encoded_seq = win
    compressed_seq, compressed_counts = compress_homopolymers(encoded_seq)
    compressed_to_raw_start = np.cumsum(np.concatenate(([0], compressed_counts)))
    compressed_to_raw_end = np.cumsum(compressed_counts) 
    largest_confident_period = _get_largest_confident_period_by_k(ksize, alpha=0.1)
    
    # calculate distance on compressed sequence
    dist = calculate_distance(compressed_seq, ksize, min(largest_confident_period, max_dist))
    log_dist = np.log(dist) # distance >= 1

    # calculate smoothness score using rolling window
    n = len(log_dist)
    if n < score_vision_size:
        return np.zeros(n, dtype=np.int32)
    smoothness_score = np.zeros(n, dtype=np.int32)
    # create sliding window view without copying
    windows = sliding_window_view(
        np.pad(log_dist, (score_vision_size//2, score_vision_size//2), mode='reflect'),
        window_shape=score_vision_size
    )
    # calculate median of each window
    rolling_median = np.median(windows, axis=1)
    # calculate MAD
    abs_dev = np.abs(windows - rolling_median[:, np.newaxis])
    rolling_mad = np.median(abs_dev, axis=1)
    # avoid division by zero, if rolling_median is 0, set robust_cv to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        robust_cv = np.where(rolling_median != 0, 
                            (rolling_mad * 1.4826) / rolling_median, 
                            0.0)
    ### smoothness_score = 1.0 / (robust_cv + 0.01) # TODO, this metric decays too quickly, consider using a more gentle metric
    alpha = 0.01
    smoothness_score = 1.0 / (robust_cv + alpha) * alpha * 100
    smoothness_score = np.nan_to_num(smoothness_score, nan=0.0)
    smoothness_score = np.round(smoothness_score).astype(np.int32)

    # record smoothness distribution
    distribution = np.bincount(smoothness_score, minlength = 101)
    with open(f"{job_dir}/stats/smoothness_distribution_{ksize}.txt", 'a') as fi:
        writer = csv.writer(fi, delimiter='\t', lineterminator='\n')
        writer.writerow(distribution)
    
    smoothness_score = np.nan_to_num(smoothness_score, nan=0.0)
    
    above_threshold = smoothness_score > min_smoothness

    # add False at the edge
    padded = np.pad(above_threshold, (1, 1), 'constant', constant_values=False)
    # find the start and end of the candidate regions
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]     # switch 0 -> 1, 0-based coordinates (closed interval)
    ends = np.where(diff == -1)[0] - 1  # switch 1 -> 0, 0-based coordinates (closed interval)
    
    # use raw sequence
    dist = calculate_distance(encoded_seq, ksize, min(largest_confident_period, max_dist)) # distance >= 1 have meaningful periodicity
    windows = sliding_window_view(
        np.pad(dist, (score_vision_size//2, score_vision_size//2), mode='reflect'),
        window_shape=score_vision_size
    )
    # calculate median of each window
    rolling_median = np.median(windows, axis=1)
    starts = compressed_to_raw_start[starts]
    ends = compressed_to_raw_end[ends]

    ###print(f"rgn {[(int(s), int(e)) for s, e in zip(starts, ends)]}")

    # split regions by rolling_median changes (numba accelerated)
    RELATIVE_THRESHOLD = 0.3  # relative change to avoid over-splitting
    MIN_CONSECUTIVE_CHANGES = 2  # require at least 2 consecutive positions with changes to avoid over-splitting
    if len(starts) > 0:
        # estimate max output size: worst case is each region splits into many pieces
        # use len(starts) * 10 as a safe upper bound
        max_output_size = max(len(starts) * 10, 100)
        new_starts_arr, new_ends_arr, count = split_regions_by_rolling_median(
            starts, ends, rolling_median, max_output_size, RELATIVE_THRESHOLD, MIN_CONSECUTIVE_CHANGES
        )
        if count == max_output_size:
            logger.warning(f"max_output_size is too small, increasing to {max_output_size * 100}")
            max_output_size = max(len(starts) * 100, 100)
            new_starts_arr, new_ends_arr, count = split_regions_by_rolling_median(
                starts, ends, rolling_median, max_output_size, RELATIVE_THRESHOLD, MIN_CONSECUTIVE_CHANGES
            )
        # extract only valid regions
        starts = new_starts_arr[:count]
        ends = new_ends_arr[:count]
    else:
        # no regions to split
        starts = np.array([], dtype=np.int64)
        ends = np.array([], dtype=np.int64)
    
   # call candidate regions
    for start_idx, end_idx in zip(starts, ends):
        ###print(f"rgn {start_idx}-{end_idx}")
        if start_idx >= len(dist) or end_idx < 0:
            continue
        
        region_mask = slice(start_idx, end_idx + 1)
        region_dist = dist[region_mask]
        med_score = -1
        region_dist = region_dist[(region_dist != 1) & (~np.isnan(region_dist))] # remove 1, nan and duplicates
        if len(region_dist) == 0:
            continue
        mode_dist = get_mode(region_dist)
        if mode_dist is None:
            continue
        mode_dist_freq = (region_dist == mode_dist).sum() / len(region_dist)
        secondary_mode_dist: int | None = None
        secondary_mode_dist_freq: float = 0.0
        if mode_dist_freq <= 0.5:
            region_dist = region_dist[region_dist != mode_dist]
            secondary_mode_dist = get_mode(region_dist)
            if secondary_mode_dist is not None:
                secondary_mode_dist_freq = (region_dist == secondary_mode_dist).sum() / len(region_dist)
        
        if mode_dist_freq + secondary_mode_dist_freq <= 0.5:
            continue

        period_tuple = [(int(mode_dist), int(end_idx - start_idx + 1))]
        if secondary_mode_dist is not None:
            period_tuple.append((int(secondary_mode_dist), int(end_idx - start_idx + 1)))

        ###print(f"rgn {start_idx}-{end_idx}, mode_distance {mode_dist}, distance {region_dist}")
        
        piece_rgns.append([
            win_id,
            chrom,
            start_idx,       # relative coordinates, 0-based, closed interval
            end_idx,         # relative coordinates, 0-based, closed interval
            [ksize],         # list of ksizes
            med_score,
            period_tuple  # list of mode distances and length
        ])

    # chaining piece regions
    chaining_width = ksize * 2
    chained_rgns: List[Region] = []
    for rgn in piece_rgns:
        if not chained_rgns:
            chained_rgns.append(rgn)
            continue
        last_rgn = chained_rgns[-1]
        # check overlap
        if rgn[2] - chaining_width <= last_rgn[3] and rgn[6][0][0] == last_rgn[6][0][0] and rgn[6][0][0] >= 5 and last_rgn[6][0][0] >= 5:  # overlapped and have same period
            merged_start = min(last_rgn[2], rgn[2])
            merged_end = max(last_rgn[3], rgn[3])
            merged_score = max(last_rgn[5], rgn[5])  # keep the max score
            merged_period = merge_period_tuples(last_rgn[6], rgn[6])
            chained_rgns[-1] = [
                last_rgn[0],
                last_rgn[1],
                merged_start,
                merged_end,
                last_rgn[4] + rgn[4],  # merge ksize lists
                merged_score,
                merged_period
            ]
        else:
            chained_rgns.append(rgn)

    # recover to raw sequence
    raw_rgns = [rgn for rgn in chained_rgns if rgn[3] - rgn[2] + 1 > rgn[6][0][0] / 10.0]

    ###for rgn in raw_rgns:
    ###    print(f"---- raw_rgns {rgn}")

    return raw_rgns

def call_on_encoded(
    job_dir: str, 
    win: Tuple[int, str, int, int, np.ndarray],
    ksize: int, 
    max_dist: int, 
    score_vision_size: int, 
    min_smoothness: int
    ) -> List[Region]:
    """
    Call the raw regions for the given sequence
    Input:
        job_dir: str, job directory path
        win: Tuple[int, str, int, int, np.ndarray]
        ksize: int
        max_dist: int, maximum distance to call regions
        score_vision_size: int, window length to compute smoothness score
        min_smoothness: int, minimum smoothness score to call regions
    Output:
        raw_rgns: List[Region]
    """
    raw_rgns: List[Region] = []
    piece_rgns: List[Region] = []
    win_id, chrom, win_start, win_end, encoded_seq = win
    largest_confident_period = _get_largest_confident_period_by_k(ksize, alpha=0.1)

    # calculate distance on encoded sequence
    dist = calculate_distance(encoded_seq, ksize, min(largest_confident_period, max_dist))
    log_dist = np.log(dist) # distance >= 1

    # calculate smoothness score using rolling window
    n = len(log_dist)
    if n < score_vision_size:
        return np.zeros(n, dtype=np.int32)
    smoothness_score = np.zeros(n, dtype=np.int32)
    # create sliding window view without copying
    windows = sliding_window_view(
        np.pad(log_dist, (score_vision_size//2, score_vision_size//2), mode='reflect'),
        window_shape=score_vision_size
    )
    # calculate median of each window
    rolling_median = np.median(windows, axis=1)
    # calculate MAD
    abs_dev = np.abs(windows - rolling_median[:, np.newaxis])
    rolling_mad = np.median(abs_dev, axis=1)
    # avoid division by zero, if rolling_median is 0, set robust_cv to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        robust_cv = np.where(rolling_median != 0, 
                            (rolling_mad * 1.4826) / rolling_median, 
                            0.0)
    ### smoothness_score = 1.0 / (robust_cv + 0.01) # TODO, this metric decays too quickly, consider using a more gentle metric
    alpha = 0.1
    smoothness_score = 1.0 / (robust_cv + alpha) * alpha * 100
    smoothness_score = np.nan_to_num(smoothness_score, nan=0.0)
    smoothness_score = np.round(smoothness_score).astype(np.int32)

    # record smoothness distribution
    distribution = np.bincount(smoothness_score, minlength = 101)
    with open(f"{job_dir}/stats/smoothness_distribution_{ksize}.txt", 'a') as fi:
        writer = csv.writer(fi, delimiter='\t', lineterminator='\n')
        writer.writerow(distribution)
    
    smoothness_score = np.nan_to_num(smoothness_score, nan=0.0)
    
    above_threshold = smoothness_score > min_smoothness

    # add False at the edge
    padded = np.pad(above_threshold, (1, 1), 'constant', constant_values=False)
    # find the start and end of the candidate regions
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]     # switch 0 -> 1, 0-based coordinates (closed interval)
    ends = np.where(diff == -1)[0] - 1  # switch 1 -> 0, 0-based coordinates (closed interval)
    
    # split regions by rolling_median changes (numba accelerated)
    RELATIVE_THRESHOLD = 0.3  # relative change to avoid over-splitting
    MIN_CONSECUTIVE_CHANGES = 2  # require at least 2 consecutive positions with changes to avoid over-splitting
    if len(starts) > 0:
        # estimate max output size: worst case is each region splits into many pieces
        # use len(starts) * 10 as a safe upper bound
        max_output_size = max(len(starts) * 10, 100)
        new_starts_arr, new_ends_arr, count = split_regions_by_rolling_median(
            starts, ends, rolling_median, max_output_size, RELATIVE_THRESHOLD, MIN_CONSECUTIVE_CHANGES
        )
        if count == max_output_size:
            logger.warning(f"max_output_size is too small, increasing to {max_output_size * 100}")
            max_output_size = max(len(starts) * 100, 100)
            new_starts_arr, new_ends_arr, count = split_regions_by_rolling_median(
                starts, ends, rolling_median, max_output_size, RELATIVE_THRESHOLD, MIN_CONSECUTIVE_CHANGES
            )
        # extract only valid regions
        starts = new_starts_arr[:count]
        ends = new_ends_arr[:count]
    else:
        # no regions to split
        starts = np.array([], dtype=np.int64)
        ends = np.array([], dtype=np.int64)
    
    # call candidate regions
    for start_idx, end_idx in zip(starts, ends):
        if start_idx >= len(above_threshold) or end_idx < 0:
            continue
        
        region_mask = slice(start_idx, end_idx + 1)
        region_scores = smoothness_score[region_mask]
        region_distances = dist[region_mask]
        med_score = np.nanmedian(region_scores)
        mode_distance = get_mode(region_distances)
        
        piece_rgns.append([
            win_id,
            chrom,
            start_idx,       # relative coordinates, 0-based, closed interval
            end_idx,         # relative coordinates, 0-based, closed interval
            [ksize],         # list of ksizes
            med_score,
            [(mode_distance, int(end_idx - start_idx + 1))]  # list of mode distances and length
        ])


    # chaining piece regions
    chaining_width = ksize * 2
    chained_rgns: List[Region] = []
    for rgn in piece_rgns:
        if not chained_rgns:
            chained_rgns.append(rgn)
            continue
        last_rgn = chained_rgns[-1]
        # check overlap
        if rgn[2] - chaining_width <= last_rgn[3] and rgn[6][0][0] == last_rgn[6][0][0] and rgn[6][0][0] >= 5 and last_rgn[6][0][0] >= 5:  # overlapped and have same period
            merged_start = min(last_rgn[2], rgn[2])
            merged_end = max(last_rgn[3], rgn[3])
            merged_score = max(last_rgn[5], rgn[5])  # keep the max score
            merged_period = merge_period_tuples(last_rgn[6], rgn[6])
            chained_rgns[-1] = [
                last_rgn[0],
                last_rgn[1],
                merged_start,
                merged_end,
                last_rgn[4] + rgn[4],  # merge ksize lists
                merged_score,
                merged_period
            ]
        else:
            chained_rgns.append(rgn)

    # recover to raw sequence
    raw_rgns.extend([rgn for rgn in chained_rgns if rgn[3] - rgn[2] + 1 > rgn[6][0][0] / 10.0])

    return raw_rgns

def call_raw_rgns(
    job_dir: str, 
    win: Tuple[int, str, int, int, np.ndarray], 
    ksizes: List[int], 
    max_dist: int, 
    score_vision_size: int, 
    min_smoothness: int
    ) -> List[Region]:
    """
    Call the raw regions for the given sequence
    Input:
        job_dir: str, job directory path
        win: Tuple[int, str, int, int, np.ndarray]
        ksizes: List[int]
        max_dist: int, maximum distance to call regions
        score_vision_size: int, window length to compute smoothness score
        min_smoothness: int, minimum smoothness score to call regions
    Output:
        raw_rgns: List[Region]
    """
    raw_rgns: List[Region] = []

    for ksize in ksizes:
        #t: List[Region] = call_on_compressed(job_dir, win, ksize, max_dist, score_vision_size, min_smoothness)
        #raw_rgns.extend(t)
        t: List[Region] = call_on_encoded(job_dir, win, ksize, max_dist, score_vision_size, min_smoothness)
        raw_rgns.extend(t)

    return raw_rgns

# Step 2: merge overlapped regions
def merge_rgns(rgns: List[Region], include_offset: bool = False) -> List[Region]:
    """
    Merge overlapped regions into nonoverlapped regions
    Input:
        rgns: List[Region]
        include_offset: bool, whether to include the offset in the merge (based on the period)
    Output:
        merged_rgns: List[Region]
    """
    merged_rgns: List[Region] = []
    for rgn in rgns:
        if not merged_rgns:
            merged_rgns.append(rgn)
            continue
            
        last_rgn = merged_rgns[-1]

        # determine if the two regions are overlapped or contained, and have same period
        is_contained = (rgn[2] >= last_rgn[2]) and (rgn[3] <= last_rgn[3])
        need_merge = is_contained
        if not need_merge: # not contained
            is_overlapped = (rgn[2] - rgn[6][0][0] <= last_rgn[3]) if include_offset else (rgn[2] <= last_rgn[3])
            if is_overlapped: # but overlapped
                last_periods = {p for p, _ in last_rgn[6]}
                rgn_periods = {p for p, _ in rgn[6]}
                is_period_equal = bool(last_periods & rgn_periods)
                need_merge = is_period_equal

        if need_merge:  # overlapped and have same period
            merged_start = min(last_rgn[2], rgn[2])
            merged_end = max(last_rgn[3], rgn[3])
            merged_ksize = sorted(list(set(last_rgn[4] + rgn[4])))      # merge ksize lists
            merged_score = max(last_rgn[5], rgn[5])  # keep the max score
            merged_period = merge_period_tuples(last_rgn[6], rgn[6])
            merged_rgns[-1] = [
                last_rgn[0],
                last_rgn[1],
                merged_start,
                merged_end,
                merged_ksize,
                merged_score,
                merged_period
            ]
        else:
            merged_rgns.append(rgn)

    return merged_rgns

def calculate_edit_distance_between_motifs(m1: np.ndarray, m2: np.ndarray) -> int:
    """
    Calculate the edit distance between two motifs
    Input:
        motif_i: np.ndarray
        motif_j: np.ndarray
    Output:
        edit_distance: int
    """
    # m1 is the longer motif
    if len(m1) < len(m2):
        m1, m2 = m2, m1

    score_array, band_argmax_j, trace_M, trace_I, trace_D = banded_dp_align(
        seq = m1,
        motif = m2,
        band_width = len(m2),
        align_to_end = True
    )
                
    state: int= np.argmax(score_array[len(m1) - 1, :]) # 0 -> M, 1 -> I, 2 -> D
    best_j: int = band_argmax_j[len(m1) - 1, state]

    # calculate edit distance from traceback
    ops, _, _ = traceback_banded_roll_motif(
        trace_M = trace_M,
        trace_I = trace_I,
        trace_D = trace_D,
        best_i = len(m1),
        best_j = best_j,
        m = len(m2),
        seq = m1,
        motif = m2
    )
    # count edit operations: 'X' (mismatch), 'I' (insertion), 'D' (deletion)
    edit_distance = sum(1 for op in ops if op in ['X', 'I', 'D'])
    return edit_distance

# Step 3: remove concatemer
def remove_concatemer(rgns: List[Region], encoded_seq: np.ndarray) -> List[Region]:
    """
    Remove the concatemer motif from the candidates
    Input:
        rgns: List[Region]
        encoded_seq: np.ndarray
    Output:
        rgns: List[Region]
    """
    for idx, rgn in enumerate(rgns):
        if len(rgn[6]) <= 1:
            continue

        # get periods
        period_to_tuple: Dict[int, Tuple[int, int]] = {p: (p, l) for p, l in rgn[6]}
        period_dedup: List[int] = sorted(period_to_tuple.keys(), reverse=True)
        
        collapsed: Dict[int, int] = {p: p for p in period_dedup}
        # Cache motifs to avoid repeated get_most_frequent_kmer calls
        motif_cache: Dict[int, np.ndarray | None] = {}
        included_seq: np.ndarray = encoded_seq[rgn[2] : rgn[3] + 1] # 0-based, closed interval
        
        for i in range(len(period_dedup)): # longer motif
            p_i = period_dedup[i]
            # early exit if already collapsed
            if collapsed[p_i] != p_i:
                continue
            
            for j in range(len(period_dedup) - 1, i, -1): # shorter motif
                p_j = period_dedup[j]

                # skip if the shorter motif is not more frequent than the longer motif
                if period_to_tuple[p_i][1] >= period_to_tuple[p_j][1]:
                    continue

                # get or compute motif_i with caching
                if p_i not in motif_cache:
                    motif_i, _ = get_most_frequent_kmer(included_seq, p_i)
                    motif_cache[p_i] = motif_i
                else:
                    motif_i = motif_cache[p_i]
                if motif_i is None:
                    continue

                # skip if not possible concatemer
                if not is_possible_concatemer(p_i, p_j):
                    continue
                
                if p_j not in motif_cache:
                    motif_j, _ = get_most_frequent_kmer(included_seq, p_j)
                    motif_cache[p_j] = motif_j
                else:
                    motif_j = motif_cache[p_j]
                
                if motif_j is None:
                    continue

                edit_distance = calculate_edit_distance_between_motifs(motif_i, motif_j)
                
                if edit_distance <= 0.0 * p_i: # TODO 0.4 0.2 any better threshold?, could be good with 0.2
                    collapsed[p_i] = p_j
                    break

        # reconstruct tuples with original lengths, sum lengths for same period
        period_dict: Dict[int, int] = {}
        for p, l in rgn[6]:
            final_period = collapsed[p]
            if final_period in period_dict:
                period_dict[final_period] += l
            else:
                period_dict[final_period] = l
        
        # write collapsed periods back (was missing — rgn[6] never updated)
        new_periods: List[Tuple[int, int]] = sorted(
            [(p, l) for p, l in period_dict.items()],
            key=lambda x: x[0],
        )
        if len(new_periods) == 0:
            new_periods = list(rgn[6])
        rgns[idx][6] = new_periods

    return rgns

# Step 4: polish borders
def polish_rgns(rgns: List[Region], win: Tuple[int, str, int, int, np.ndarray]) -> List[Region]:
    """
    Polish the candidates with more accurate period
    Input:
        rgns: List[window_id, chrom, start, end, k, score, period]
        win: Tuple[int, str, int, int, np.ndarray], win_id, chrom, start, end, encoded_seq
    Output:
        polished_rgns: List[window_id, chrom, start, end, k, score, period]
    """
    win_id, chrom, _, _, encoded_seq = win
    seq_len = len(encoded_seq)
    MAX_CONTEXT_LEN = 5000   # the maximum length of the context to extend

    ref_motifs: List[str] = []
    for rgn in rgns:
        best_period, _ = get_most_frequent_period_tuple(rgn[6])
        included_end: int = rgn[3] + 1
        included_start: int = max(0, rgn[2] - best_period)
        included_seq: np.ndarray = encoded_seq[included_start : included_end]
        ref_motif, _ = get_most_frequent_kmer(included_seq, best_period)
        ref_motifs.append(ref_motif)

    idx: int = 0
    polished_rgns: List[Region] = []
    cur_start: int | None = None
    cur_end: int | None = None
    cur_period: List[Tuple[int, int]] = []
    while idx < len(rgns):
        # get basic information
        rgn: Region = rgns[idx]
        next_rgn: Region | None = rgns[idx + 1] if idx + 1 < len(rgns) else None

        # extract period values (first element of tuple) for get_mode
        cur_period: List[Tuple[int, int]] = merge_period_tuples(cur_period, rgn[6])
        best_period, _ = get_most_frequent_period_tuple(cur_period)
        MAX_EXTEND_LEN: int = min(best_period * 500, MAX_CONTEXT_LEN)
        MAX_INIT_LEN: int = best_period * 1

        # polish start
        if cur_start is None:
            ref_motif = ref_motifs[idx]
            extend_start = max(0, rgn[2] - MAX_EXTEND_LEN)
            extend_end = rgn[3]
            confirmed_tr_end = rgn[3] - rgn[2] + 1 # 1-based
            extend_len = extend_border(encoded_seq[extend_start : extend_end + 1][::-1], ref_motif[::-1], confirmed_tr_end, None)
            cur_start = max(0, rgn[3] - extend_len + 1)
            ###print(f" $$ ----- rgn {rgn[1]}:{rgn[2]}-{rgn[3]} cur_start: {cur_start}, extend_len: {extend_len}")
        
        # polish end
        included_start = cur_start
        included_end = rgn[3]
        if rgn[3] - cur_start + 1 > best_period * 100: # if the region is too long, use nearest 100 periods to get motif (hack for satellites)
            included_start = included_end - best_period * 100
        included_seq = encoded_seq[included_start : included_end + 1]
        ref_motif, _ = get_most_frequent_kmer(included_seq, best_period)
        # TODO does this situation exist?
        if ref_motif is None:
            ref_motif = encoded_seq[rgn[3] + 1 - best_period : rgn[3] + 1]
            ###logger.warning(f"ref_motif is None! included_seq = {included_seq}, best_period = {best_period} for region {rgn[1]}:{rgn[2]}-{rgn[3]}") # TODO ???
        
        if next_rgn is None:  # the last region
            extend_start = included_start
            extend_end = min(rgn[3] + MAX_EXTEND_LEN, seq_len - 1)
            confirmed_tr_end = rgn[3] - extend_start + 1 # 1-based
            extend_seq = encoded_seq[extend_start : extend_end + 1]
            extend_len = extend_border(extend_seq, ref_motif, confirmed_tr_end, None)
            cur_end = extend_start + extend_len - 1
            polished_rgns.append([win_id, chrom, cur_start, cur_end, rgn[4], rgn[5], cur_period])
            cur_start, cur_end, cur_period = None, None, []
            idx += 1
            continue

        have_downstream = False
        if next_rgn[2] - rgn[3] - 1 <= MAX_CONTEXT_LEN:  # not the last region, consider the coordinates and the motif similarity of the next region             
            edit_distance = calculate_edit_distance_between_motifs(ref_motif, ref_motifs[idx + 1])
            is_highly_similar = edit_distance <= 0.4 * max(len(ref_motif), len(ref_motifs[idx + 1]))  # TODO: how to decide the threshold?
            if is_highly_similar:
                have_downstream = True
        
        """
        if have_downstream:
            extend_end = min(next_rgn[3], seq_len - 1)
            extend_start = max(included_start, rgn[3] - 10 * best_period) # TODO ???
            extend_seq = encoded_seq[extend_start : extend_end + 1]
            confirmed_tr_end = rgn[3] - extend_start + 1 # 1-based
            extend_len = extend_border(extend_seq, ref_motif, True, confirmed_tr_end)

            if extend_start + extend_len >= next_rgn[2]: # jump to next region
                idx += 1
                continue
        
        # do not have downstream, extend the border
        extend_end = min(rgn[3] + MAX_EXTEND_LEN, seq_len - 1)
        extend_start = max(included_start, rgn[3] - 10 * best_period) # TODO ???
        extend_seq = encoded_seq[extend_start : extend_end + 1]
        extend_len = extend_border(extend_seq, ref_motif, False, None)
        """
        extend_end = min(rgn[3] + MAX_EXTEND_LEN, seq_len - 1)
        extend_start = max(included_start, rgn[3] - 10 * best_period) # TODO ???
        extend_seq = encoded_seq[extend_start : extend_end + 1]
        confirmed_tr_end = rgn[3] - extend_start + 1 # 1-based
        compared_tr_end = None
        if have_downstream:
            compared_tr_end = min(next_rgn[3] - extend_start + 1, len(extend_seq)) # 1-based

        extend_len = extend_border(extend_seq, ref_motif, confirmed_tr_end, compared_tr_end)

        ###print(f" $$ rgn {rgn[1]}:{rgn[2]}-{rgn[3]} extend_seq range: {extend_start}-{extend_end}, ref_motif: {decode_array_to_seq(ref_motif)}")
        ###print(f" $$ rgn {rgn[1]}:{rgn[2]}-{rgn[3]} cur_end: {extend_start + extend_len}, extend_len: {extend_len}")
        # record region
        if extend_start + extend_len < next_rgn[2]:
            cur_end = extend_start + extend_len - 1
            polished_rgns.append([win_id, chrom, cur_start, cur_end, rgn[4], rgn[5], cur_period])
            ###print(f" $$ rgn {rgn[1]}:{rgn[2]}-{rgn[3]} end extending")
            cur_start, cur_end, cur_period = None, None, []
        else:
            ###print(f" $$ rgn {rgn[1]}:{rgn[2]}-{rgn[3]} still extending")
            pass
        idx += 1

    return polished_rgns

def extend_border(
    encoded_seq: np.ndarray, 
    encoded_motif: np.ndarray, 
    confirmed_tr_end: int,
    compared_tr_end: int | None = None
) -> int:
    """
    Extend the border of the motif to the sequence, with the minimum score to link the region
    Input:
        encoded_seq: np.ndarray (target)
        encoded_motif: np.ndarray (query)
        confirmed_tr_end: int | None, the confirmed tr end position (1-based)
        compared_tr_end: int | None, the compared tr end position (1-based)
    Output:
        extend_len: int, the length to extend the border
    """
    seq_len: int = len(encoded_seq)
    if seq_len == 0:
        return 0

    have_downstream = compared_tr_end is not None
    if have_downstream:
        compare_row = compared_tr_end - 1
    else:
        compare_row = -1  # Numba: no Optional; -1 disables downstream early-exit branch

    # align
    score_array, _, _, _, _ = banded_dp_align(
        seq = encoded_seq,
        motif = encoded_motif,
        band_width = MAX_PERIOD,
        align_to_end = False, ### have_downstream, # TODO
        anchor_row = confirmed_tr_end - 1,
        compare_row = compare_row,
    )
    
    if have_downstream and score_array[compared_tr_end - 1, :].max() >= score_array[confirmed_tr_end - 1, :].max():
        ###score = score_array[-1, :].max()
        end_i = seq_len
    else: # get the maximum score in the sequence
        ###score = score_array[:, 0].max()
        end_i = score_array[:, 0].argmax() + 1

    return end_i

def get_motif_and_seq(
    rgns: List[Region], 
    win: Tuple[int, str, int, int, np.ndarray]
) -> List[RegionWithMotifAndSeq]:
    """
    Get the candidate motif and sequence of the regions
    Input:
        rgns: List[Region]
        win: Tuple[int, str, int, int, np.ndarray], win_id, chrom, win_start, win_end, encoded_seq
    Output:
        rgns_with_motif_and_seq: List[RegionWithMotifAndSeq]
    """
    win_id, chrom, win_start, win_end, encoded_seq = win
    rgns_with_motif_and_seq: List[RegionWithMotifAndSeq] = []

    for rgn in rgns:
        rgn_seq: np.ndarray = encoded_seq[rgn[2] : rgn[3] + 1]
        rgn_motifs: List[str] = []
        for motif_len, _ in rgn[6]:  # extract period value (first element of tuple)
            rgn_motif, _ = get_most_frequent_kmer(rgn_seq, motif_len)
            if rgn_motif is not None:
                rgn_motifs.append(rgn_motif)
        # coordinates: 0-based (closed interval) -> 1-based (closed interval)
        rgns_with_motif_and_seq.append([
            win_id, 
            chrom, 
            rgn[2] + win_start + 1, 
            rgn[3] + win_start + 1, 
            rgn[4], 
            rgn[5], 
            rgn[6], 
            decode_array_to_seq(rgn_seq), 
            [decode_array_to_seq(motif) for motif in rgn_motifs]
        ])
    
    return rgns_with_motif_and_seq


"""
#
# codes for identfying repeats across windows
#
"""
# main entry
def identify_region_across_windows(results: List[str], seq_win_size: int, seq_ovlp_size: int) -> List[str]:
    """
    Process the region across multiple windows
    Input:
        results: List[str], the path of the polished candidates file
        seq_window_size: int, the size of the sequence window
        seq_overlap_size: int, the overlap size of the sequence window
    Output:
        List[str], the path of the linked regions file
    """
    csv.field_size_limit(1024 * 1024 * 1024)  # set as 1 GB to avoid overflow
    active_rgn: Optional[RegionWithMotifAndSeq] = None
    across_win_rgns: List[RegionWithMotifAndSeq] = []
    prev_result: Optional[str] = None
    prev_chrom: Optional[str] = None
    is_prev_chrom_start: bool = False
    prev_start_rgns, prev_mid_rgns, prev_end_rgns = [], [], []
    for result in results:
        # read regions in current window
        cur_start_rgns, cur_mid_rgns, cur_end_rgns, cur_st_mid_rgn, cur_mid_end_rgn = [], [], [], None, None
        with open(result, 'r', newline='', encoding='utf-8') as fi:
            reader = csv.reader(fi, delimiter='\t')
            rows = list(reader)

        if len(rows) == 0:
            if prev_result:
                with open(prev_result, 'w', newline='', encoding='utf-8') as fi:
                    writer = csv.writer(fi, delimiter='\t', lineterminator='\n')
                    if is_prev_chrom_start:
                        writer.writerows(prev_start_rgns)
                    writer.writerows(prev_mid_rgns)
                    writer.writerows(prev_end_rgns)
            active_rgn = None
            continue

        # read window file
        win_id = int(rows[0][0])
        win_filepath = f"{Path(result).parent.parent}/windows/window_{win_id}.tsv"
        with open(win_filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            win_rows = list(reader)
        _, chrom, win_start, _, _ = win_rows[0]
        win_start = int(win_start)
        
        is_cur_chrom_start = False
        if prev_chrom is None or prev_chrom != chrom:
            is_cur_chrom_start = True
        
        for row in rows:
            row[2], row[3] = int(row[2]), int(row[3])
            win_id, chrom, start, end, _, _, _, _, _ = row  # ksizes, score, periods, seq, motifs
            # classify regions based on position
            if start <= win_start + seq_ovlp_size and end > win_start + seq_ovlp_size:
                row[4], row[6], row[8] = (
                    ast.literal_eval(row[4]),
                    ast.literal_eval(row[6]),
                    ast.literal_eval(row[8]),
                )
                cur_st_mid_rgn = row
            elif end >= win_start + seq_win_size - seq_ovlp_size and start < win_start + seq_win_size - seq_ovlp_size:
                row[4], row[6], row[8] = (
                    ast.literal_eval(row[4]),
                    ast.literal_eval(row[6]),
                    ast.literal_eval(row[8]),
                )
                cur_mid_end_rgn = row
            elif start <= win_start + seq_ovlp_size:
                cur_start_rgns.append(row)
            elif end >= win_start + seq_win_size - seq_ovlp_size:
                cur_end_rgns.append(row)
            else:
                cur_mid_rgns.append(row)

        if active_rgn and cur_st_mid_rgn and is_overlapped(active_rgn, cur_st_mid_rgn):
            active_rgn = merge_regions(active_rgn, cur_st_mid_rgn)
        elif cur_st_mid_rgn: # cannot extend, finalize the previous one
            if active_rgn:
                across_win_rgns.append(active_rgn)
            active_rgn = cur_st_mid_rgn

        if cur_mid_end_rgn:
            if active_rgn and is_overlapped(active_rgn, cur_mid_end_rgn):
                active_rgn = merge_regions(active_rgn, cur_mid_end_rgn)
            else:
                if active_rgn:
                    across_win_rgns.append(active_rgn)
                active_rgn = cur_mid_end_rgn

        if prev_result: # skip the first window
            with open(prev_result, 'w', newline='', encoding='utf-8') as fi:
                writer = csv.writer(fi, delimiter='\t', lineterminator='\n')
                if is_prev_chrom_start:
                    writer.writerows(prev_start_rgns)
                writer.writerows(prev_mid_rgns)
                writer.writerows(prev_end_rgns)
        
        # update
        prev_start_rgns, prev_mid_rgns, prev_end_rgns = cur_start_rgns, cur_mid_rgns, cur_end_rgns
        prev_st_mid_rgn, prev_mid_end_rgn = cur_st_mid_rgn, cur_mid_end_rgn
        prev_result, prev_chrom, is_prev_chrom_start = result, chrom, is_cur_chrom_start
    
    # wrtie regions in the last window
    if prev_result:
        with open(prev_result, 'w', newline='', encoding='utf-8') as fi:
            writer = csv.writer(fi, delimiter='\t', lineterminator='\n')
            if is_prev_chrom_start:
                writer.writerows(prev_start_rgns)
            writer.writerows(prev_mid_rgns)
            writer.writerows(prev_end_rgns)
    
    # add the active region in the last window
    if active_rgn:
        across_win_rgns.append(active_rgn)
    
    # write the linked regions to file
    merged_filepath = f"{Path(results[0]).parent}/window_linked.tsv"
    with open(merged_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(across_win_rgns)
            
    return [merged_filepath] + results

def is_overlapped(rgn1: RegionWithMotifAndSeq, rgn2: RegionWithMotifAndSeq) -> bool:
    """
    Check if two regions have overlap
    Input:
        rgn1: RegionWithMotifAndSeq, [win_id, chrom, start, end, k, score, periods, seq, motifs]
        rgn2: RegionWithMotifAndSeq, [win_id, chrom, start, end, k, score, periods, seq, motifs]
    Output:
        bool
    """
    return rgn1[1] == rgn2[1] and max(rgn1[2], rgn2[2]) <= min(rgn1[3], rgn2[3])

def merge_regions(rgn1: RegionWithMotifAndSeq, rgn2: RegionWithMotifAndSeq) -> RegionWithMotifAndSeq:
    """
    Merge two regions
    Input:
        rgn1: RegionWithMotifAndSeq, [win_id, chrom, start, end, k, score, periods, seq, motifs]
        rgn2: RegionWithMotifAndSeq, [win_id, chrom, start, end, k, score, periods, seq, motifs]
    Output:
        merged_rgn: RegionWithMotifAndSeq
    """
    # make sure rgn1 is on the left
    if rgn2[2] < rgn1[2]:
        rgn1, rgn2 = rgn2, rgn1

    s1, e1 = rgn1[2], rgn1[3]
    s2, e2 = rgn2[2], rgn2[3]
    seq1, seq2 = rgn1[7], rgn2[7]

    # merge sequence
    if e2 <= e1:
        merged_seq = seq1
    else:
        overlap_len = e1 - s2 + 1
        merged_seq = seq1 + seq2[overlap_len:]
    merged_motif_lens = list(set(rgn1[6] + rgn2[6]))
    merged_motifs = list(set(rgn1[8] + rgn2[8]))
    return [
        rgn1[0],
        rgn1[1],
        min(s1, s2),
        max(e1, e2),
        rgn1[4] + rgn2[4],
        rgn1[5],
        merged_motif_lens,
        merged_seq,
        merged_motifs
    ]


"""
#
# codes for annotating repeats
#
"""
def annotate_regions(task: Tuple[str, str, str, int, float]) -> str:
    """
    Annotate the regions in one window
    Input:
        polished_rgn_path: str, the path of the polished candidates file
        output_filepath: str, the path of the annotated file
        format: str, the format of the output file, select from ["trf", "verbose"]
        min_score: int, the minimum score threshold for the region
        min_copy: float, the minimum copy number threshold for the region
    Output:
        output_filepath: str, the path of the annotated file
    """
    polished_rgn_path, output_filepath, format, min_score, min_copy = task
    csv.field_size_limit(1024 * 1024 * 1024)  # set as 1 GB to avoid overflow
    with open(polished_rgn_path, 'r', newline='', encoding='utf-8') as fi:
        reader = csv.reader(fi, delimiter='\t')
        rows = list(reader)
    output_rows: List[Any] = []

    match format:
        case "brief":
            HEADER = ["chrom", "start", "end", "period", "copyNumber",
                      "score", "motif"]
        case "trf":
            HEADER = ["chrom", "start", "end", "period", "copyNumber", 
                      "consensusSize", "percentMatches", "percentIndels", "score", "A", "C", "G", "T", "entropy", "motif", "sequence", "cigar"]
        case "bed": # bed12 + extra 12 columns
            HEADER = ["chrom", "start", "end", "motif", "pseudoScore", "strand", "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts",
                      "period", "copyNumber", "percentMatches", "percentIndels", "score", "A", "C", "G", "T", "entropy", "sequence", "cigar"]

    # start alignment for each region
    for row in rows:
        rgn_dict = {
            "win_id": row[0],
            "chrom": row[1],
            "start": int(row[2]),
            "end": int(row[3]),
            "ksizes": row[4].translate(str.maketrans("", "", "[] ")),
            "smoothness": row[5],
            "periods": ast.literal_eval(row[6]),
            "sequence": row[7],
            "motifs": ast.literal_eval(row[8])
        }
        if format == "bed":
            rgn_dict.update({
                "pseudoScore": 1000,
                "strand": ".",
                "start": int(row[2]) - 1,
                "thickStart": int(row[2]) - 1,
                "thickEnd": int(row[3]),
                "itemRgb": ".",
                "blockCount": 1,
                "blockSizes": int(row[3]) - int(row[2]),
                "blockStarts": 0
            })
        max_score: int = 0
        candidates: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, int, int, str]] = [] # score, trace_M, trace_I, trace_D, end_i, end_j, motif
        encoded_seq = encode_seq_to_array(rgn_dict["sequence"])
        seq_len = len(encoded_seq)
        for motif in rgn_dict["motifs"]:
            encoded_motif = encode_seq_to_array(motif)
            score_array, band_argmax_j, trace_M, trace_I, trace_D = banded_dp_align(
                seq = encoded_seq,
                motif = encoded_motif,
                band_width = MAX_PERIOD,
                align_to_end = True
            )
            
            score = score_array[seq_len - 1, :].max()
            state = score_array[seq_len - 1, :].argmax()
            end_i = len(encoded_seq) # 1-based
            end_j = band_argmax_j[seq_len - 1, state]
            # get motif profile
            profile = traceback_motif_profile(trace_M, trace_I, trace_D, end_i, end_j, len(motif), encoded_seq)
            # get refined motif and remove gaps
            encoded_refined_motif = profile.argmax(axis=1)
            encoded_refined_motif = encoded_refined_motif[encoded_refined_motif != 4]

            # check the periodicity of the refined motif
            is_periodic, period = is_periodic_seq(encoded_refined_motif)
            if is_periodic:
                encoded_refined_motif = encoded_refined_motif[:period] # get the minimal period
            if len(encoded_refined_motif) == 0:
                ###raise RuntimeError(f"Empty refined motif for region {row}, motif: {motif}, refined_motif: {encoded_refined_motif}")
                continue

            score_array, band_argmax_j, trace_M, trace_I, trace_D = banded_dp_align(
                seq = encoded_seq,
                motif = encoded_refined_motif,
                band_width = MAX_PERIOD,
                align_to_end = True
            )
            ###print(f"final alignment seq: {encoded_seq}, motif: {encoded_refined_motif}")
            score = int(score_array[seq_len - 1:, 0]) # TODO only match? or indel is also ok?
            end_i = seq_len # 1-based
            end_j = band_argmax_j[seq_len - 1:, 0]
            if score_array.size == 0:
                raise RuntimeError(f"Empty alignment score array for region {row}")
            if score >= max_score * SECONDARY_SCORE_RATIO:
                ###print(f"score: {score}, motif: {decode_array_to_seq(encoded_refined_motif)}, end_i: {end_i}, end_j: {end_j}")
                candidates.append((score, trace_M, trace_I, trace_D, end_i, end_j, encoded_refined_motif))
                max_score = max(max_score, score)

        if len(candidates) == 0:
            continue

        # skip the region if the score is too low
        if max_score < min_score:
            continue

        # filter and sort candidates by score
        candidates = [candidate for candidate in candidates if candidate[0] >= max_score * SECONDARY_SCORE_RATIO]
        candidates.sort(key=lambda x: (-x[0], len(x[6])))
        cur_score = 2 ** 31 - 1
        for candidate in candidates:
            score, trace_M, trace_I, trace_D, end_i, end_j, encoded_motif = candidate
            if score == cur_score:
                continue
            cur_score = score
            # get alignment operations and starting positions
            # The alignment may not start from the beginning of the motif or sequence
            # start_i: starting position in seq (1-based DP index, 0 means start from beginning)
            # start_j: starting position in motif (0-based index, 0 means start from beginning)
            ops: List[str]
            start_i: int
            start_j: int
            ops, start_i, start_j = traceback_banded_roll_motif(
                trace_M = trace_M,
                trace_I = trace_I,
                trace_D = trace_D,
                best_i = end_i,
                best_j = end_j,
                m = len(encoded_motif),
                seq = encoded_seq,
                motif = encoded_motif,
            )
        
            # Adjust motif based on starting position
            if start_j > 0:
                # Rotate motif so that alignment starts from the beginning
                adjusted_motif = np.concatenate((encoded_motif[start_j:], encoded_motif[:start_j]))
            else:
                adjusted_motif = encoded_motif

            # refine motif
            ### consensus = xxxxxxx()
        
            rgn_dict["motif"] = decode_array_to_seq(adjusted_motif)
            rgn_dict["period"] = len(adjusted_motif) # TODO
            rgn_dict["consensusSize"] = len(adjusted_motif)
            rgn_dict["score"] = score

            cigar: str = ops_to_cigar(ops, len(adjusted_motif))
            rgn_dict["cigar"] = cigar if not SKIP_CIGAR else "."
            rgn_dict["copyNumber"] = get_copy_number(cigar, rgn_dict["period"])

            if rgn_dict["copyNumber"] < min_copy:
                continue

            # calculate nucleotide composition and entropy
            if format in ["trf", "bed"]:
                rgn_dict["percentMatches"], rgn_dict["percentIndels"] = calculate_alignment_metrics(ops, len(rgn_dict["sequence"]))
                nt_comp, entropy = calculate_nucleotide_composition(rgn_dict["sequence"])
                rgn_dict["A"], rgn_dict["C"], rgn_dict["G"], rgn_dict["T"] = nt_comp["A"], nt_comp["C"], nt_comp["G"], nt_comp["T"]
                rgn_dict["entropy"] = entropy
        
            # add output row
            output_rows.append([rgn_dict[k] for k in HEADER])

    # wrtie resutls
    with open(output_filepath, 'w', newline='', encoding='utf-8') as fo:
        writer = csv.writer(fo, delimiter='\t', lineterminator='\n')
        writer.writerow(HEADER)
        writer.writerows(output_rows)
    
    return output_filepath

def encode_ops(ops: List[str]) -> np.ndarray:
    """
    Encode the operations into an integer array
    Inputs:
        ops: List[str], the operations
    Outputs:
        np.ndarray, the encoded operations
    """
    return OPS_TABLE[np.frombuffer("".join(ops).encode(), dtype=np.uint8)]

@numba.njit(cache=True)
def traceback_motif_profile(
    trace_M: np.ndarray, 
    trace_I: np.ndarray, 
    trace_D: np.ndarray, 
    best_i: int, 
    best_j: int, 
    m: int, 
    seq: np.ndarray
) -> np.ndarray:
    """
    Get the motif profile from the traceback matrix
    Inputs:
        trace_M: np.ndarray, traceback matrix for match / mismatch
        trace_I: np.ndarray, traceback matrix for insertion
        trace_D: np.ndarray, traceback matrix for deletion
        best_i: int, best index in seq
        best_j: int, best index in motif
        m: int, length of motif
        seq: np.ndarray, sequence
    Outputs:
        np.ndarray, the motif profile
    """
    i, j = best_i, best_j
    state = 0

    motif_profile = np.zeros((m, 5), dtype=np.int32)

    while i > 0: # index of seq, 1-based
        if state == 0:  # M
            prev_state = trace_M[i, j]
            motif_profile[j, seq[i - 1]] += 1
            i -= 1
            j = j - 1 if j > 0 else m - 1
            state = prev_state

        elif state == 1:  # I
            prev_state = trace_I[i, j]
            i -= 1
            state = prev_state
            motif_profile[j, 4] += 1

        elif state == 2:  # D
            prev_state = trace_D[i, j]
            j = j - 1 if j > 0 else m - 1
            state = prev_state
            ### motif_profile[j, 4] += 1

    return motif_profile

@numba.njit(cache=True)
def banded_dp_align(
    seq: np.ndarray,
    motif: np.ndarray,
    band_width: int,
    align_to_end: bool = False,
    anchor_row: int = -1,
    compare_row: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Inputs:
            seq: np.ndarray, sequence
            motif: np.ndarray, motif
            band_width: int, band width
            align_to_end: bool, whether to align to end
            anchor_row: int, 0-based anchor row; -1 if unused
            compare_row: int, 0-based compare row; -1 means no downstream / skip compare logic
        Outputs:
            score_array: np.ndarray, score array
            band_argmax_j: np.ndarray, band argmax j
            trace_M: np.ndarray, traceback matrix for match / mismatch
            trace_I: np.ndarray, traceback matrix for insertion
            trace_D: np.ndarray, traceback matrix for deletion
    """
    # score_array[i, s] = max M/I/D in band at DP row i+1; s in {0,1,2} -> M,I,D.
    # band_argmax_j[i, s] = motif column j (0-based) where that row-state max is attained
    # (first j in band scan order on ties, same as strict > updates).
    n = seq.shape[0]
    m = motif.shape[0]
    NEG_INF = -10 ** 9
    # Numba nopython cannot type i <= compare_row when compare_row may be None; use -1 sentinel.
    have_downstream = compare_row >= 0

    score_array = np.full((n, 3), NEG_INF, np.int32)
    band_argmax_j = np.full((n, 3), -1, np.int32)

    # ---- DP matrices ----
    M = np.full((n + 1, m), NEG_INF, np.int32)
    I = np.full((n + 1, m), NEG_INF, np.int32)
    D = np.full((n + 1, m), NEG_INF, np.int32)

    trace_M = np.full((n + 1, m), -1, np.int8)
    trace_I = np.full((n + 1, m), -1, np.int8)
    trace_D = np.full((n + 1, m), -1, np.int8)

    # ---- init ----
    for j in range(m):
        M[0, j] = 0

    # ---- precompute run-length of seq ----
    run_len = np.ones(n, dtype=np.int32)
    for i in range(1, n):
        if seq[i] == seq[i - 1]:
            run_len[i] = run_len[i - 1] + 1
        else:
            run_len[i] = 1

    # ---- parameters for scaling ----
    alpha = 0.5   # control the decay strength (can be adjusted)
    min_scale = 0.3  # lower bound, prevent gap too cheap

    best_score = NEG_INF

    for i in range(1, n + 1):

        j_center = (i - 1) % m
        j_start = max(0, j_center - band_width)
        j_end   = min(m, j_center + band_width + 1)

        si = seq[i - 1]
        cur_score_m = NEG_INF
        cur_score_i = NEG_INF
        cur_score_d = NEG_INF
        cur_j_m = -1
        cur_j_i = -1
        cur_j_d = -1

        # ---- current run-length ----
        rl = run_len[i - 1]

        # scale: 1 / (1 + alpha*(rl-1))
        scale = 1.0 / (1.0 + alpha * (rl - 1))
        if scale < min_scale:
            scale = min_scale

        gap_open_scaled   = int(GAP_OPEN_PENALTY * scale)
        gap_extend_scaled = int(GAP_EXTEND_PENALTY * scale)

        for j in range(j_start, j_end):

            prev_j = m - 1 if j == 0 else j - 1

            # ---- match/mismatch ----
            s = MATCH_SCORE if si == motif[j] else -MISMATCH_PENALTY

            # ---- M ----
            best_prev = M[i - 1, prev_j]
            state = 0

            v = I[i - 1, prev_j]
            if v > best_prev:
                best_prev = v
                state = 1

            v = D[i - 1, prev_j]
            if v > best_prev:
                best_prev = v
                state = 2

            M[i, j] = best_prev + s
            trace_M[i, j] = state

            # ---- I (gap in motif) ----
            open_i = M[i - 1, j] - gap_open_scaled
            ext_i  = I[i - 1, j] - gap_extend_scaled

            if open_i > ext_i:
                I[i, j] = open_i
                trace_I[i, j] = 0
            else:
                I[i, j] = ext_i
                trace_I[i, j] = 1

            # ---- D (gap in seq) ----
            open_d = M[i, prev_j] - gap_open_scaled
            ext_d  = D[i, prev_j] - gap_extend_scaled

            if open_d > ext_d:
                D[i, j] = open_d
                trace_D[i, j] = 0
            else:
                D[i, j] = ext_d
                trace_D[i, j] = 2

            # ---- record ----
            if M[i, j] > cur_score_m:
                cur_score_m = M[i, j]
                cur_j_m = j
            if I[i, j] > cur_score_i:
                cur_score_i = I[i, j]
                cur_j_i = j
            if D[i, j] > cur_score_d:
                cur_score_d = D[i, j]
                cur_j_d = j

        score_array[i - 1, 0] = cur_score_m
        band_argmax_j[i - 1, 0] = cur_j_m
        score_array[i - 1, 1] = cur_score_i
        band_argmax_j[i - 1, 1] = cur_j_i
        score_array[i - 1, 2] = cur_score_d
        band_argmax_j[i - 1, 2] = cur_j_d
        best_score = max(best_score, cur_score_m, cur_score_i, cur_score_d)

        # ---- early exit ----
        if not align_to_end:
            if have_downstream and i <= compare_row:
                continue
            # if have downstream, and the score of the compare row is higher than the score of the anchor row, break
            if have_downstream and score_array[anchor_row, :].max() <= score_array[compare_row, :].max():
                break
            # if already compute the compare row, and no likelihood to exceed the best score, break
            if (n - i) * MATCH_SCORE + max(cur_score_m, cur_score_i, cur_score_d) <= best_score:
                break
            """# if already compute the anchor row, and no likelihood to exceed the anchor row score, break
            if anchor_row is not None and i + 1 >= anchor_row and (n - i) * MATCH_SCORE + max(cur_score_m, cur_score_i, cur_score_d) <= score_array[anchor_row, :].max():
                break"""
    
    return (
        score_array,
        band_argmax_j,
        trace_M, trace_I, trace_D,
    )

def traceback_banded_roll_motif(
    trace_M: np.ndarray, trace_I: np.ndarray, trace_D: np.ndarray,
    best_i: int, best_j: int,
    m: int,
    seq: np.ndarray,
    motif: np.ndarray,
) -> Tuple[List[str], int, int]:
    """
    Inputs:
        trace_M: np.ndarray, traceback matrix for match / mismatch
        trace_I: np.ndarray, traceback matrix for insertion
        trace_D: np.ndarray, traceback matrix for deletion
        best_i: int, best index in seq
        best_j: int, best index in motif
        m: int, length of motif
        seq: np.ndarray, target sequence
        motif: np.ndarray, query motif
    Outputs:
        ops: List[str], atomic operations: '=', 'X', 'I', 'D', '/'
        start_i: int, starting position in seq (1-based DP index; 0 means start from beginning)
        start_j: int, starting position in motif (0-based index)
    Notes:
        state: 0=M (diagonal), 1=I (gap in motif), 2=D (gap in seq)
    """
    i, j = best_i, best_j
    state = 0

    ops: List[str] = []

    while i > 0: # index of seq, 1-based
        if state == 0:  # M
            prev_state = trace_M[i, j]
            ops.append("=" if seq[i - 1] == motif[j] else "X")
            i -= 1
            j = j - 1 if j > 0 else m - 1
            state = prev_state

        elif state == 1:  # I
            prev_state = trace_I[i, j]
            i -= 1
            state = prev_state
            ops.append("I")

        elif state == 2:  # D
            prev_state = trace_D[i, j]
            j = j - 1 if j > 0 else m - 1
            state = prev_state
            ops.append("D")

    ops.reverse()

    # i, 1-based DP index into seq; j, 0-based motif index before the first aligned motif base.
    start_i = i
    start_j = (j + 1) % m if m > 0 else 0

    return ops, start_i, start_j



"""
#
# merge outputs
#
"""
def merge_outputs(job_dir: str, rgn_filepaths: List[str]) -> str:
    """
    Merge sorted separate output files into a single sorted output file.
    Uses streaming merge for memory efficiency.
    Inputs:
        job_dir : str, job directory
        rgn_filepaths : List[str], list of output filepaths (first is linked file)
    Outputs:
        output_filepath : str, path to the merged output file
    """
    csv.field_size_limit(1024 * 1024 * 1024)
    
    if not rgn_filepaths:
        return None
    
    linked_filepath = rgn_filepaths[0]
    other_filepaths = rgn_filepaths[1:]
    
    # Get header from first file
    header = None
    with open(linked_filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
    
    def get_key(row):
        """Get sort key: (chrom, start, end)"""
        return (row[0], int(row[1]), int(row[2]))
    
    # File iterator wrapper for heap merge
    class FileIterator:
        def __init__(self, filepath, file_idx):
            self.filepath = filepath
            self.file_idx = file_idx
            self.file = open(filepath, 'r', newline='', encoding='utf-8')
            self.reader = csv.reader(self.file, delimiter='\t')
            next(self.reader)  # skip header
            try:
                self.current_row = next(self.reader)
                self.key = get_key(self.current_row)
            except StopIteration:
                self.current_row = None
                self.key = None
        
        def __lt__(self, other):
            if self.key is None:
                return False
            if other.key is None:
                return True
            return self.key < other.key
        
        def next_row(self):
            try:
                self.current_row = next(self.reader)
                self.key = get_key(self.current_row)
                return True
            except StopIteration:
                self.current_row = None
                self.key = None
                self.file.close()
                return False
    
    # Initialize all file iterators
    iterators = []
    try:
        # Linked file iterator
        it = FileIterator(linked_filepath, 0)
        if it.current_row is not None:
            iterators.append(it)
        
        # Other file iterators
        for idx, filepath in enumerate(other_filepaths, 1):
            if not Path(filepath).exists():
                continue
            it = FileIterator(filepath, idx)
            if it.current_row is not None:
                iterators.append(it)
    except Exception as e:
        logging.warning(f"Error initializing file iterators: {e}")
        for it in iterators:
            if hasattr(it, 'file'):
                it.file.close()
        return None
    
    # Multi-way merge using heap
    heapq.heapify(iterators)
    output_filepath = f"{job_dir}/final_results.tsv"
    total_rows = 0
    
    with open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter='\t', lineterminator='\n')
        writer.writerow(header)
        
        while iterators:
            # Get smallest row
            smallest = heapq.heappop(iterators)
            writer.writerow(smallest.current_row)
            total_rows += 1
            
            # Load next row from this file
            if smallest.next_row():
                heapq.heappush(iterators, smallest)
    
    return output_filepath

"""
#
# main function for scanning the genome
#
"""
def run_scan(cfg: dict[str, Any]) -> None:
    """
    Run the scan function.
    Inputs:
        cfg : dict[str, Any], configuration dictionary
    Outputs:
        None
    """
    JOB_DIR = cfg["job_dir"]
    # Set global variables for alignment functions
    global MATCH_SCORE, MISMATCH_PENALTY, GAP_OPEN_PENALTY, GAP_EXTEND_PENALTY
    MATCH_SCORE = cfg["match_score"]
    MISMATCH_PENALTY = cfg["mismatch_penalty"]
    GAP_OPEN_PENALTY = cfg["gap_open_penalty"]
    GAP_EXTEND_PENALTY = cfg["gap_extend_penalty"]
    global MAX_PERIOD, MIN_LEN, SECONDARY_SCORE_RATIO
    MAX_PERIOD = cfg["max_period"]
    MIN_LEN = cfg["min_score"] / MATCH_SCORE
    SECONDARY_SCORE_RATIO = cfg["secondary"]
    global SKIP_CIGAR
    SKIP_CIGAR = cfg["skip_cigar"]
    global ENCODE_TABLE
    ENCODE_TABLE = np.full(256, -1, dtype=np.int8)
    ENCODE_TABLE[ord('A')] = 0
    ENCODE_TABLE[ord('C')] = 1
    ENCODE_TABLE[ord('G')] = 2
    ENCODE_TABLE[ord('T')] = 3
    global OPS_TABLE
    OPS_TABLE = np.full(256, -1, dtype=np.int8)
    OPS_TABLE[ord("=")] = 0
    OPS_TABLE[ord("M")] = 0
    OPS_TABLE[ord("X")] = 1
    OPS_TABLE[ord("I")] = 2
    OPS_TABLE[ord("D")] = 3
    
    if MATCH_SCORE < 0 or MISMATCH_PENALTY < 0 or GAP_OPEN_PENALTY < 0 or GAP_EXTEND_PENALTY < 0:
        msg = f"Match score, mismatch penalty, gap open penalty, and gap extend penalty must be non-negative integers"
        logger.error(f"ERROR: {msg}")
        raise ValueError(msg)
    if cfg["seq_win_size"] <= cfg["seq_ovlp_size"]:
        msg = f"Sequence window size must be greater than overlap size"
        logger.error(f"ERROR: {msg}")
        raise ValueError(msg)

    START_TIME: float = time.time()
    logger.info(f"{cfg["threads"]} cores are used")

    # read input fasta file
    fasta: List[SeqRecord] = list(read_fasta(cfg["input"]))
    logger.info(f"Finished reading fasta file: {cfg["input"]}")

    # split fasta file into windows
    Path(JOB_DIR + "/windows").mkdir(parents=True, exist_ok=True)
    win_filepaths: List[str] = split_fasta_by_window(fasta, cfg["seq_win_size"], cfg["seq_ovlp_size"], JOB_DIR)
    logger.info(f"Finished splitting windows: n = {len(win_filepaths)} windows created")

    # call non-overlapped tandem repeat regions
    Path(JOB_DIR + "/stats").mkdir(parents=True, exist_ok=True)
    Path(JOB_DIR + "/raw_rgns").mkdir(parents=True, exist_ok=True)
    Path(JOB_DIR + "/polished_rgns").mkdir(parents=True, exist_ok=True)
    tasks: List[Tuple[str, str, List[int], int, int, int]] = [(JOB_DIR, win_filepath, cfg["ksize"], cfg["max_period"], cfg["rolling_win_size"], cfg["min_smoothness"]) for win_filepath in win_filepaths]
    try:
        with Pool(processes = cfg["threads"]) as pool:
            results: List[str] = []
            for result in tqdm(
                pool.imap(call_regions, tasks, chunksize = 2),
                total = len(tasks),
                desc ="calling repeats"
            ):
                if result is None:
                    logger.warning("One task returned None, may indicate failure")
                elif isinstance(result, str):
                    results.append(result)
                else:
                    logger.warning(f"Unexpected result type: {type(result)}, skipping")
        logger.info(f"Finished calling tandem repeat regions")
    except Exception as e:
        msg = f"Error in calling regions: {e}"
        logger.error(msg, exc_info=True)
        raise RuntimeError(msg)

    # get regions that are across multiple windows
    results: List[str] = identify_region_across_windows(results, cfg["seq_win_size"], cfg["seq_ovlp_size"])
    logger.info(f"Finished identifying regions across multiple windows")

    # annotate regions
    Path(JOB_DIR + "/annotated_rgns").mkdir(parents=True, exist_ok=True)
    output_filepaths: List[str] = [f"{JOB_DIR}/annotated_rgns/{Path(rgn_path).name}" for rgn_path in results]
    tasks: List[Tuple[str, str, str, int]] = [(rgn_path, output_filepath, cfg["format"], cfg["min_score"], cfg["min_copy"]) for rgn_path, output_filepath in zip(results, output_filepaths)]
    try:
        with Pool(processes = cfg["threads"]) as pool:
            results = []
            for result in tqdm(
                pool.imap(annotate_regions, tasks, chunksize = 1),
                total = len(tasks),
                desc = "annotating repeats"
            ):
                if result is None:
                    logger.warning("One annotation task returned None, may indicate failure")
                results.append(result)
    except Exception as e:
        msg = f"Error in annotating regions: {e}"
        logger.error(msg, exc_info=True)
        raise RuntimeError(msg)
    
    # generate final result (sorted) - merge already sorted files
    output_filepath = merge_outputs(JOB_DIR, output_filepaths)
    if output_filepath:
        logger.info(f"Merged results")
    
    # Copy final results file
    final_results_src = f"{JOB_DIR}/final_results.tsv"
    final_results_dst = f"{cfg["prefix"]}.tsv"
    shutil.copy2(final_results_src, final_results_dst)
    logger.info(f"Generated final annotation: {final_results_dst}")
    
    END_TIME: float = time.time()
    TIME_USED: float = round(END_TIME - START_TIME, 2)
    logger.info(f"Time used: {TIME_USED:.2f} seconds")

    # make stats and report
    is_empty = False
    with open(final_results_src) as f:
        f.readline()  # header
        if not f.readline():
            is_empty = True
    if not cfg["skip_report"] and not is_empty:            
        # import utils
        from vampire._report_utils import make_stats, make_report
        from importlib.resources import files
        # add data
        cfg["time_used"] = TIME_USED
        cfg["subcommand"] = "scan"
        # make stats and report
        data: dict[str, Any] = make_stats(cfg)
        make_report(JOB_DIR, str(files("vampire.resources").joinpath("scan_web_summary_template.html")), data)
        # copy web summary file
        web_summary_src = f"{JOB_DIR}/web_summary.html"
        web_summary_dst = f"{cfg["prefix"]}.web_summary.html"
        shutil.copy2(web_summary_src, web_summary_dst)
        logger.info(f"Generated web summary: {web_summary_dst}")
    else:
        if is_empty:
            logger.info(f"No tandem repeat detected. Skipping report generation")
        else:
            logger.info(f"Skipping report generation")

    logger.info("Bye.")
    
    # copy log file
    shutil.copy2(f"{JOB_DIR}/log.log", f"{cfg["prefix"]}.log")