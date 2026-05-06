from __future__ import annotations
from typing import Any, Iterator

# data processing
import numpy as np
import polars as pl
from math import sqrt
from Bio import SeqRecord, SeqIO
from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view  # rolling median
# multiprocessing
from tqdm import tqdm
from multiprocessing import Pool
# basic packages for file operations, logging
import ast
import csv
import time
import logging
import heapq
import shutil
import tempfile
from pathlib import Path
# numba packages for speed optimization
import numba

from vampire._report_utils import(
    ops_to_cigar,
    get_copy_number,
    calculate_alignment_metrics,
    calculate_nucleotide_composition
)
from vampire._utils import(
    encode_seq_to_array,
    decode_array_to_seq,
    compress_homopolymers,
    encode_array_to_int,
    decode_int_to_array,
    canonicalize_motif,
    is_periodic_seq,
)

logger = logging.getLogger(__name__)

# type definitions, the coordinates are 0-based and closed interval
Region = list[Any] # [win_id, chrom, start, end, ksizes, score, periods] : [int, str, int, int, list[int], float, list[tuple(int, int)]]
RegionWithMotifAndSeq = list[Any] # [win_id, chrom, start, end, ksizes, score, periods, seq, motifs] : [int, str, int, int, list[int], float, list[tuple(int, int)], str, list[str]]



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

def split_fasta_by_window(fasta: Iterator[SeqRecord], seq_win_size: int, seq_ovlp_size: int, job_dir: str) -> list[str]:
    """
    Split fasta file into windows of the given size and overlap
    Input:
        fasta: Iterator[SeqRecord]
        seq_win_size: int
        seq_ovlp_size: int
        job_dir: str
    Output:
        window_filepaths: list[str]
        for each window, the format is [window_id, chrom, start, end, seq], seq is upper case DNA string
    """
    if seq_ovlp_size >= seq_win_size:
        raise ValueError("Overlap size must be less than window size")

    window_filepaths: list[str] = []
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
def get_most_frequent_kmer(encoded_seq: np.ndarray, k: int) -> tuple[np.ndarray | None, int]:
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

        # fallback: if all k-mers are periodic, return the most frequent k-mer
        if mf_canonical_kmer_int is None:
            if not counts:
                return None, 0
            mf_canonical_kmer_int = max(counts.items(), key=lambda x: x[1])[0]
            mf_count = counts[mf_canonical_kmer_int]

        # decode back to sequence
        mf_array = decode_int_to_array(mf_canonical_kmer_int, k)
        
    else:
        counts = defaultdict(int)
        mf_canonical_kmer_tuple: tuple[int, ...] | None = None
        mf_canonical_array: np.ndarray | None = None  # cache array to avoid final conversion
        mf_count: int = 0
        low_complex_cache: dict[tuple[int, ...], int] = {}  # cache period values
        canonical_cache: dict[bytes, tuple[tuple[int, ...], np.ndarray]] = {}  # bytes -> (tuple, array)
        
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

        # fallback: if all k-mers are periodic, return the most frequent k-mer
        if mf_canonical_kmer_tuple is None:
            if not counts:
                return None, None
            else:
                mf_canonical_kmer_tuple = max(counts.items(), key=lambda x: x[1])[0]
                mf_count = counts[mf_canonical_kmer_tuple]
                mf_canonical_array = np.array(mf_canonical_kmer_tuple, dtype=np.int8)
    
        mf_array = mf_canonical_array

    return mf_array, mf_count

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
        align_to_end = True,
        anchor_row = -1, # no anchor
        compare_row = -1, # no compare
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


"""
#
# codes for calling repeats
#
"""
# main entry
def call_regions(task: tuple[str, list[int], int, int, int]) -> str:
    """
    Call the nonoverlapped tandem repeat regions for the given sequence
    Input:
        task: tuple[str, list[int], int, int, int, int], includes:
            window_filepath: str, window file path
            ksizes: list[int], ksize list
            max_dist: int, maximum distance to call regions
            score_vision_size: int, window length to compute smoothness score
            min_smoothness: int, minimum smoothness score to call regions
    Output:
        result: str, raw region file path, the format is:
            list[window_id, chrom, start, end, k, score, period], start and end are 1-based and include the borders
    """
    window_filepath, ksizes, max_dist, score_vision_size, min_smoothness = task
    # read window information
    csv.field_size_limit(2 * 1024 * 1024 * 1024)  # set as 2 GB to avoid overflow
    with open(window_filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            win_id, chrom, win_start, win_end, seq = row
    win_start, win_end = int(win_start), int(win_end)
    encoded_seq = encode_seq_to_array(seq)
    win = (win_id, chrom, win_start, win_end, encoded_seq)
    del seq

    # call raw regions
    raw_rgns: list[Region] = call_raw_rgns(win, ksizes, max_dist, score_vision_size, min_smoothness)
    logger.debug(f"window {win_id}: call_raw_rgns finished")

    # if no raw regions found, return None
    if not raw_rgns:
        return None
    
    # sort by coordinates (start from small to large, end from large to small)
    raw_rgns.sort(key=lambda x: (x[2], -x[3]))

    # merge overlapped or contained raw regions, merge k into list, use max score
    merged_rgns: list[Region] = merge_rgns(raw_rgns, include_offset=True)
    del raw_rgns
    logger.debug(f"window {win_id}: merge_rgns finished")

    # return None if no merged regions
    if not merged_rgns:
        return None

    # remove concatemer motifs
    merged_rgns = remove_concatemer(merged_rgns, encoded_seq)
    logger.debug(f"window {win_id}: remove_concatemer finished")

    # write merged raw regions to file
    output_filepath = f"{JOB_DIR}/raw_rgns/window_{win_id}.tsv"
    with open(output_filepath, 'w', newline='', encoding='utf-8') as fi:
        writer = csv.writer(fi, delimiter='\t', lineterminator='\n')
        writer.writerows(merged_rgns)
    logger.debug(f"window {win_id}: write_raw_rgns finished")

    # polish region borders
    polished_rgns: list[Region] = polish_rgns(merged_rgns, win)
    del merged_rgns
    logger.debug(f"window {win_id}: polish_rgns finished")

    # sort by coordinates
    polished_rgns.sort(key=lambda x: (x[2], -x[3]))

    # merge overlapped polished regions
    merged_rgns: list[Region] = merge_rgns(polished_rgns, include_offset=False)
    del polished_rgns
    logger.debug(f"window {win_id}: merge_rgns finished")

    # filter polished regions that are too short
    final_rgns: list[Region] = list(filter(lambda x: x[3] - x[2] + 1 >= MIN_LEN, merged_rgns))
    del merged_rgns
    logger.debug(f"window {win_id}: filter_rgns finished, total {len(final_rgns)} regions left")
    
    # remove concatemer motifs
    final_rgns = remove_concatemer(final_rgns, encoded_seq)
    logger.debug(f"window {win_id}: polished regions remove_concatemer finished")

    # get motif and sequence, transform 0-based coordinates (closed interval) to 1-based global coordinates (closed interval)
    rgns_with_motif_and_seq: list[RegionWithMotifAndSeq] = get_motif_and_seq(final_rgns, win)
    del final_rgns, win
    logger.debug(f"window {win_id}: get_motif_and_seq finished")

    # write polished regions to file
    output_filepath = f"{JOB_DIR}/polished_rgns/window_{win_id}.tsv"
    with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(rgns_with_motif_and_seq)

    logger.debug(f"window {win_id}: call_region finished")

    return output_filepath

def get_mode(nums: list[int]) -> int | None:
    """
    Get mode from a list of elements using numpy
    Input:
        nums: list[int]
    Output:
        mode: int|None, return None if no mode is found
    """
    if isinstance(nums, np.ndarray):
        nums = nums[~np.isnan(nums)]
    else: # list[int]
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

def get_most_frequent_period_tuple(period_tuples: list[tuple[int, int]]) -> tuple[int, int]:
    """
    Get the most frequent period tuple from a list of period tuples based on the length
    Input:
        period_tuples: list[tuple[int, int]]
    Output:
        tuple[int, int]: the most frequent period tuple
    """
    mf_p, mf_l = None, 0
    for p, l in period_tuples:
        if l > mf_l:
            mf_p = p
            mf_l = l
    return mf_p, mf_l

def merge_period_tuples(period_tuples1: list[tuple[int, int]], period_tuples2: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Merge two lists of period tuples, sorted by length from long to short
    Input:
        period_tuples1: list[tuple[int, int]]
        period_tuples2: list[tuple[int, int]]
    Output:
        list[tuple[int, int]]: the merged period tuples, sorted by period
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
    return sorted([(p, l) for p, l in period_dict.items()], key=lambda x: -x[1])

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
) -> tuple[np.ndarray, np.ndarray, int]:
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
    win: tuple[int, str, int, int, np.ndarray],
    ksize: int, 
    max_dist: int, 
    score_vision_size: int, 
    min_smoothness: int
    ) -> list[Region]:
    """
    Call the raw regions for the given sequence
    Input:
        win: tuple[int, str, int, int, np.ndarray]
        ksize: int
        max_dist: int, maximum distance to call regions
        score_vision_size: int, window length to compute smoothness score
        min_smoothness: int, minimum smoothness score to call regions
    Output:
        raw_rgns: list[Region]
    """
    raw_rgns: list[Region] = []
    piece_rgns: list[Region] = []
    win_id, chrom, win_start, win_end, encoded_seq = win
    compressed_seq, compressed_counts = compress_homopolymers(encoded_seq)
    compressed_to_raw_start = np.cumsum(np.concatenate(([0], compressed_counts)))
    compressed_to_raw_end = np.cumsum(compressed_counts) 
    largest_confident_period = _get_largest_confident_period_by_k(ksize, alpha=0.1)
    
    # calculate distance on compressed sequence
    dist = calculate_distance(compressed_seq, ksize, min(largest_confident_period, max_dist))
    del compressed_seq, compressed_counts
    log_dist = np.log(dist) # distance >= 1
    del dist

    # calculate smoothness score using rolling window
    n = len(log_dist)
    if n < score_vision_size:
        return []
    smoothness_score = np.zeros(n, dtype=np.int32)
    # create sliding window view without copying
    windows = sliding_window_view(
        np.pad(log_dist, (score_vision_size//2, score_vision_size//2), mode='reflect'),
        window_shape=score_vision_size
    )
    del log_dist
    # calculate median of each window
    rolling_median = np.median(windows, axis=1)
    # calculate MAD
    abs_dev = np.abs(windows - rolling_median[:, np.newaxis])
    rolling_mad = np.median(abs_dev, axis=1)
    del windows, abs_dev
    # avoid division by zero, if rolling_median is 0, set robust_cv to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        robust_cv = np.where(rolling_median != 0,
                            (rolling_mad * 1.4826) / rolling_median,
                            0.0)
    alpha = 0.01 # control the speed of decay
    smoothness_score = 1.0 / (robust_cv + alpha) * alpha * 100
    smoothness_score = np.nan_to_num(smoothness_score, nan=0.0)
    smoothness_score = np.round(smoothness_score).astype(np.int32)
    del rolling_mad, rolling_median, robust_cv

    # record smoothness distribution
    """
    distribution = np.bincount(smoothness_score, minlength = 101)
    with open(f"{JOB_DIR}/stats/smoothness_distribution_{ksize}.txt", 'a') as fi:
        writer = csv.writer(fi, delimiter='\t', lineterminator='\n')
        writer.writerow(distribution)
    """
    
    smoothness_score = np.nan_to_num(smoothness_score, nan=0.0)
    
    above_threshold = smoothness_score > min_smoothness

    # add False at the edge
    padded = np.pad(above_threshold, (1, 1), 'constant', constant_values=False)
    # find the start and end of the candidate regions
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]     # switch 0 -> 1, 0-based coordinates (closed interval)
    ends = np.where(diff == -1)[0] - 1  # switch 1 -> 0, 0-based coordinates (closed interval)
    del smoothness_score, above_threshold, padded, diff

    # use raw sequence
    dist = calculate_distance(encoded_seq, ksize, min(largest_confident_period, max_dist)) # distance >= 1 have meaningful periodicity
    windows = sliding_window_view(
        np.pad(dist, (score_vision_size//2, score_vision_size//2), mode='reflect'),
        window_shape=score_vision_size
    )
    # calculate median of each window
    rolling_median = np.median(windows, axis=1)
    del windows
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
        mode_dist_count: int = (region_dist == mode_dist).sum()
        mode_dist_freq: float = mode_dist_count / float(len(region_dist))
        secondary_mode_dist: int | None = None
        secondary_mode_dist_freq: float = 0.0
        if mode_dist_freq <= 0.5:
            region_dist = region_dist[region_dist != mode_dist]
            secondary_mode_dist = get_mode(region_dist)
            if secondary_mode_dist is not None:
                secondary_mode_dist_count: int = (region_dist == secondary_mode_dist).sum()
                secondary_mode_dist_freq: float = secondary_mode_dist_count / float(len(region_dist))
        
        if mode_dist_freq + secondary_mode_dist_freq <= 0.5:
            continue

        period_tuple = [(int(mode_dist), int(mode_dist_count))]
        if secondary_mode_dist is not None:
            period_tuple.append((int(secondary_mode_dist), int(secondary_mode_dist_count)))

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
    chained_rgns: list[Region] = []
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

    return raw_rgns

def call_on_encoded(
    win: tuple[int, str, int, int, np.ndarray],
    ksize: int, 
    max_dist: int, 
    score_vision_size: int, 
    min_smoothness: int
    ) -> list[Region]:
    """
    Call the raw regions for the given sequence
    Input:
        win: tuple[int, str, int, int, np.ndarray]
        ksize: int
        max_dist: int, maximum distance to call regions
        score_vision_size: int, window length to compute smoothness score
        min_smoothness: int, minimum smoothness score to call regions
    Output:
        raw_rgns: list[Region]
    """
    raw_rgns: list[Region] = []
    piece_rgns: list[Region] = []
    win_id, chrom, win_start, win_end, encoded_seq = win
    largest_confident_period = _get_largest_confident_period_by_k(ksize, alpha=0.1)

    # calculate distance on encoded sequence
    dist = calculate_distance(encoded_seq, ksize, min(largest_confident_period, max_dist))
    log_dist = np.log(dist) # distance >= 1

    # calculate smoothness score using rolling window
    n = len(log_dist)
    if n < score_vision_size:
        return []
    smoothness_score = np.zeros(n, dtype=np.int32)
    # create sliding window view without copying
    windows = sliding_window_view(
        np.pad(log_dist, (score_vision_size//2, score_vision_size//2), mode='reflect'),
        window_shape=score_vision_size
    )
    del log_dist
    # calculate median of each window
    rolling_median = np.median(windows, axis=1)
    # calculate MAD
    abs_dev = np.abs(windows - rolling_median[:, np.newaxis])
    rolling_mad = np.median(abs_dev, axis=1)
    del windows, abs_dev
    # avoid division by zero, if rolling_median is 0, set robust_cv to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        robust_cv = np.where(rolling_median != 0,
                            (rolling_mad * 1.4826) / rolling_median,
                            0.0)
    alpha = 0.1
    smoothness_score = 1.0 / (robust_cv + alpha) * alpha * 100
    smoothness_score = np.nan_to_num(smoothness_score, nan=0.0)
    smoothness_score = np.round(smoothness_score).astype(np.int32)
    del rolling_mad, robust_cv

    # record smoothness distribution
    distribution = np.bincount(smoothness_score, minlength = 101)
    with open(f"{JOB_DIR}/stats/smoothness_distribution_{ksize}.txt", 'a') as fi:
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
    del rolling_median
    
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
    chained_rgns: list[Region] = []
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
    win: tuple[int, str, int, int, np.ndarray], 
    ksizes: list[int], 
    max_dist: int, 
    score_vision_size: int, 
    min_smoothness: int
    ) -> list[Region]:
    """
    Call the raw regions for the given sequence
    Input:
        win: tuple[int, str, int, int, np.ndarray]
        ksizes: list[int]
        max_dist: int, maximum distance to call regions
        score_vision_size: int, window length to compute smoothness score
        min_smoothness: int, minimum smoothness score to call regions
    Output:
        raw_rgns: list[Region]
    """
    raw_rgns: list[Region] = []

    for ksize in ksizes:
        t: list[Region] = call_on_compressed(win, ksize, max_dist, score_vision_size, min_smoothness)
        raw_rgns.extend(t)
        t: list[Region] = call_on_encoded(win, ksize, max_dist, score_vision_size, min_smoothness)
        raw_rgns.extend(t)

    return raw_rgns

# Step 2: merge overlapped regions
def merge_rgns(rgns: list[Region], include_offset: bool = False) -> list[Region]:
    """
    Merge overlapped regions into nonoverlapped regions
    Input:
        rgns: list[Region]
        include_offset: bool, whether to include the offset in the merge (based on the period)
    Output:
        merged_rgns: list[Region]
    """
    merged_rgns: list[Region] = []
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


# Step 3: remove concatemer
def remove_concatemer(rgns: list[Region], encoded_seq: np.ndarray) -> list[Region]:
    """
    Remove the concatemer motif from the candidates
    Input:
        rgns: list[Region]
        encoded_seq: np.ndarray
    Output:
        rgns: list[Region]
    """
    for idx, rgn in enumerate(rgns):
        if len(rgn[6]) <= 1:
            continue

        # get periods
        period_to_tuple: dict[int, tuple[int, int]] = {p: (p, l) for p, l in rgn[6]}
        period_dedup: list[int] = sorted(period_to_tuple.keys(), reverse=True)
        
        collapsed: dict[int, int] = {p: p for p in period_dedup}
        # Cache motifs to avoid repeated get_most_frequent_kmer calls
        motif_cache: dict[int, np.ndarray | None] = {}
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
                
                if edit_distance <= 0.2 * p_i: # TODO any better threshold?
                    collapsed[p_i] = p_j
                    break

        # reconstruct tuples with original lengths, sum lengths for same period
        period_dict: dict[int, int] = {}
        l_sum: int = 0
        for p, l in rgn[6]:
            l_sum += l
            final_period = collapsed[p]
            if final_period in period_dict:
                period_dict[final_period] += l
            else:
                period_dict[final_period] = l
        
        # write collapsed periods back (was missing — rgn[6] never updated)
        period_tuples: list[tuple[int, int]] = sorted(
            [(p, l) for p, l in period_dict.items()],
            key=lambda x: -x[1],
        )
        l_cur: int = 0
        new_period_tuples: list[tuple[int, int]] = []
        for p, l in period_tuples:
            l_cur += l
            new_period_tuples.append((p, l))
            if l_cur / float(l_sum) >= 0.8:
                break

        if len(new_period_tuples) >= 5:
            new_period_tuples = new_period_tuples[:5]
        rgns[idx][6] = new_period_tuples

    return rgns

# Step 4: polish borders
def polish_rgns(rgns: list[Region], win: tuple[int, str, int, int, np.ndarray]) -> list[Region]:
    """
    Polish the candidates with more accurate period
    Input:
        rgns: list[window_id, chrom, start, end, k, score, period]
        win: tuple[int, str, int, int, np.ndarray], win_id, chrom, start, end, encoded_seq
    Output:
        polished_rgns: list[window_id, chrom, start, end, k, score, period]
    """
    win_id, chrom, _, _, encoded_seq = win
    seq_len = len(encoded_seq)
    MAX_CONTEXT_LEN = 5000   # the maximum length of the context to extend

    ref_motifs: list[str] = []
    for rgn in rgns:
        best_period, _ = get_most_frequent_period_tuple(rgn[6])
        included_end: int = rgn[3] + 1
        included_start: int = max(0, rgn[2] - best_period)
        included_seq: np.ndarray = encoded_seq[included_start : included_end]
        ref_motif, _ = get_most_frequent_kmer(included_seq, best_period) # ref_motif = none if the region is too short or contains invalid bases
        ref_motifs.append(ref_motif)

    idx: int = 0
    polished_rgns: list[Region] = []
    cur_start: int | None = None
    cur_end: int | None = None
    cur_period: list[tuple[int, int]] = []
    cur_extended_idx: int = 0
    while idx < len(rgns):
        # get basic information
        rgn: Region = rgns[idx]
        next_rgn: Region | None = rgns[idx + 1] if idx + 1 < len(rgns) else None

        # extract period values (first element of tuple) for get_mode
        cur_period: list[tuple[int, int]] = merge_period_tuples(cur_period, rgn[6])
        best_period, _ = get_most_frequent_period_tuple(cur_period)
        MAX_EXTEND_LEN: int = min(best_period * 500, MAX_CONTEXT_LEN)
        MAX_INIT_LEN: int = best_period * 1

        # polish start
        if cur_start is None:
            ref_motif = ref_motifs[idx]
            if ref_motif is None:
                idx += 1 
                cur_period = [] # clear cur_period
                cur_extended_idx = 0
                continue
            extend_start = max(0, rgn[2] - MAX_EXTEND_LEN)
            extend_end = rgn[3]
            confirmed_tr_end = rgn[3] - rgn[2] + 1 # 1-based
            extend_len = extend_border(encoded_seq[extend_start : extend_end + 1][::-1], ref_motif[::-1], confirmed_tr_end, None)
            cur_start = max(0, rgn[3] - extend_len + 1)
        
        # skip if already computed
        if cur_start is not None and next_rgn is not None and cur_extended_idx >= next_rgn[2]:
            idx += 1
            continue
        
        # polish end
        included_start = cur_start
        included_end = rgn[3]
        if rgn[3] - cur_start + 1 > best_period * 100: # if the region is too long, use nearest 100 periods to get motif (hack for satellites)
            included_start = included_end - best_period * 100
        included_seq = encoded_seq[included_start : included_end + 1]
        ref_motif, _ = get_most_frequent_kmer(included_seq, best_period)
        if ref_motif is None:
            ref_motif = encoded_seq[rgn[3] + 1 - best_period : rgn[3] + 1]
        
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
        if next_rgn[2] - rgn[3] - 1 <= MAX_CONTEXT_LEN and ref_motifs[idx + 1] is not None:  # not the last region, consider the coordinates and the motif similarity of the next region
            edit_distance = calculate_edit_distance_between_motifs(ref_motif, ref_motifs[idx + 1])
            is_highly_similar = edit_distance <= 0.4 * max(len(ref_motif), len(ref_motifs[idx + 1]))  # TODO: how to decide the threshold?
            if is_highly_similar:
                have_downstream = True
        
        extend_end = min(rgn[3] + MAX_EXTEND_LEN, seq_len - 1)
        extend_start = max(included_start, rgn[3] - 10 * best_period)
        extend_seq = encoded_seq[extend_start : extend_end + 1]
        confirmed_tr_end = rgn[3] - extend_start + 1 # 1-based
        compared_tr_end = None
        if have_downstream:
            compared_tr_end = min(next_rgn[3] - extend_start + 1, len(extend_seq)) # 1-based

        extend_len = extend_border(extend_seq, ref_motif, confirmed_tr_end, compared_tr_end)
        cur_extended_idx = extend_start + extend_len - 1

        # record region
        if cur_extended_idx < next_rgn[2]:
            cur_end = cur_extended_idx
            polished_rgns.append([win_id, chrom, cur_start, cur_end, rgn[4], rgn[5], cur_period])
            cur_start, cur_end, cur_period = None, None, []
        else:
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
        compare_row = -1

    # align
    score_array, _, _, _, _ = banded_dp_align(
        seq = encoded_seq,
        motif = encoded_motif,
        band_width = MAX_PERIOD,
        align_to_end = False,
        anchor_row = confirmed_tr_end - 1,
        compare_row = compare_row,
    )

    if have_downstream and score_array[compared_tr_end - 1, :].max() >= score_array[confirmed_tr_end - 1, :].max():
        end_i = compared_tr_end
    else: # get the maximum score in the sequence
        end_i = score_array[:, 0].argmax() + 1

    return end_i

def get_motif_and_seq(
    rgns: list[Region], 
    win: tuple[int, str, int, int, np.ndarray]
) -> list[RegionWithMotifAndSeq]:
    """
    Get the candidate motif and sequence of the regions
    Input:
        rgns: list[Region]
        win: tuple[int, str, int, int, np.ndarray], win_id, chrom, win_start, win_end, encoded_seq
    Output:
        rgns_with_motif_and_seq: list[RegionWithMotifAndSeq]
    """
    win_id, chrom, win_start, win_end, encoded_seq = win
    rgns_with_motif_and_seq: list[RegionWithMotifAndSeq] = []

    for rgn in rgns:
        rgn_seq: np.ndarray = encoded_seq[rgn[2] : rgn[3] + 1]
        rgn_motifs: list[str] = []
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
def identify_region_across_windows(results: list[str], seq_win_size: int, seq_ovlp_size: int) -> list[str]:
    """
    Process the region across multiple windows
    Input:
        results: list[str], the path of the polished candidates file
        seq_window_size: int, the size of the sequence window
        seq_overlap_size: int, the overlap size of the sequence window
    Output:
        list[str], the path of the linked regions file
    """
    csv.field_size_limit(2 * 1024 * 1024 * 1024)  # set as 1 GB to avoid overflow
    active_rgn: RegionWithMotifAndSeq | None = None
    across_win_rgns: list[RegionWithMotifAndSeq] = []
    prev_result: str | None = None
    prev_chrom: str | None = None
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

        if active_rgn and cur_st_mid_rgn and _is_overlapped(active_rgn, cur_st_mid_rgn):
            active_rgn = _merge_regions(active_rgn, cur_st_mid_rgn)
        elif cur_st_mid_rgn: # cannot extend, finalize the previous one
            if active_rgn:
                across_win_rgns.append(active_rgn)
            active_rgn = cur_st_mid_rgn

        if cur_mid_end_rgn:
            if active_rgn and _is_overlapped(active_rgn, cur_mid_end_rgn):
                active_rgn = _merge_regions(active_rgn, cur_mid_end_rgn)
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
    cot = 1
    for rgn in across_win_rgns:
        merged_filepath = f"{Path(results[0]).parent}/window_linked_{cot}.tsv"
        with open(merged_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(rgn)
        cot += 1

    new_filepath = [f"{Path(results[0]).parent}/window_linked_{cot}.tsv" for cot in range(1, cot)]
    
    return new_filepath + results

def _is_overlapped(rgn1: RegionWithMotifAndSeq, rgn2: RegionWithMotifAndSeq) -> bool:
    """
    Check if two regions have overlap
    Input:
        rgn1: RegionWithMotifAndSeq, [win_id, chrom, start, end, k, score, periods, seq, motifs]
        rgn2: RegionWithMotifAndSeq, [win_id, chrom, start, end, k, score, periods, seq, motifs]
    Output:
        bool
    """
    return rgn1[1] == rgn2[1] and max(rgn1[2], rgn2[2]) <= min(rgn1[3], rgn2[3])

def _merge_regions(rgn1: RegionWithMotifAndSeq, rgn2: RegionWithMotifAndSeq) -> RegionWithMotifAndSeq:
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
    # merge periods
    merged_period_tuples = merge_period_tuples(rgn1[6], rgn2[6])
    # get top accumulate 80% periods
    top_period_tuples = []
    l_sum: int = sum(l for _, l in merged_period_tuples)
    l_cur: int = 0
    for p, l in merged_period_tuples:
        l_cur += l
        top_period_tuples.append((p, l))
        if l_cur / float(l_sum) >= 0.8:
            break
    if len(top_period_tuples) > 5: # only keep the top 5 periods
        top_period_tuples = top_period_tuples[:5]
    motif_len_list = [p for p, _ in top_period_tuples]
    # merge motifs
    merged_motifs = list(set(rgn1[8] + rgn2[8]))
    top_motifs = [m for m in merged_motifs if len(m) in motif_len_list]

    return [
        f"{rgn1[0]},{rgn2[0]}",
        rgn1[1],
        min(s1, s2),
        max(e1, e2),
        rgn1[4] + rgn2[4],
        rgn1[5],
        top_period_tuples,
        merged_seq,
        top_motifs
    ]


"""
#
# codes for annotating repeats
#
"""
def annotate_regions(task: tuple[str, str, str, int, float]) -> str:
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
    csv.field_size_limit(2 * 1024 * 1024 * 1024)  # set as 1 GB to avoid overflow
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
    
    JOB_DIR: Path = Path(output_filepath).parent

    with open(output_filepath, 'w', newline='', encoding='utf-8') as fo:
        # write header
        rows: list = []
        writer = csv.writer(fo, delimiter='\t', lineterminator='\n')
        writer.writerow(HEADER)

        with open(polished_rgn_path, 'r', newline='', encoding='utf-8') as fi:
            reader = csv.reader(fi, delimiter='\t')

            # start alignment for each region
            for row in reader:
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
                        "start": int(row[2]),
                        "thickStart": int(row[2]) - 1,
                        "thickEnd": int(row[3]),
                        "itemRgb": ".",
                        "blockCount": 1,
                        "blockSizes": int(row[3]) - int(row[2]),
                        "blockStarts": 0
                    })
                max_score: int = 0
                candidates: list[tuple[int, np.ndarray, str]] = [] # score, encoded_motif, refined_motif
                encoded_seq = encode_seq_to_array(rgn_dict["sequence"])
                seq_len = len(encoded_seq)
                # memory limit for DP matrices per segment (backward pass dominant)
                # backward pass allocates: M/I/D (int32) + trace_M/I/D (int8) + score_array + band_argmax_j
                # total ≈ 15 * seg_len * motif_len bytes
                MAX_MATRIX_BYTES: int = 3 * 1024 * 1024 * 1024 # here is about 3GB for each thread

                if not rgn_dict["motifs"]:
                    continue

                MAX_MOTIF_LEN: int = max(len(motif) for motif in rgn_dict["motifs"])
                SEG_LEN: int = max(MAX_MATRIX_BYTES // (MAX_MOTIF_LEN * 15), MAX_MOTIF_LEN * 2)
                if SEG_LEN > seq_len:
                    SEG_LEN = seq_len

                subregion: list[tuple[int, int]] = []
                subregion_pre: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

                STEP_LEN: int = SEG_LEN - MAX_MOTIF_LEN
                if STEP_LEN <= 0 or seq_len <= SEG_LEN:
                    subregion = [(0, seq_len)]
                else:
                    i = 0
                    while i < seq_len:
                        end = min(i + SEG_LEN, seq_len)
                        # merge short trailing segment into previous one
                        if end - i < MAX_MOTIF_LEN and subregion:
                            subregion[-1] = (subregion[-1][0], seq_len)
                            break
                        subregion.append((i, end))
                        i += STEP_LEN
                n_sub: int = len(subregion)

                # dispatch to single- or multi-segment annotator
                if n_sub == 1:
                    tmp_rows = annotate_single_segment(
                        rgn_dict, encoded_seq, JOB_DIR, HEADER,
                        format, min_score, min_copy,
                    )
                else:
                    tmp_rows = annotate_multiple_segment(
                        rgn_dict, encoded_seq, subregion, JOB_DIR, HEADER,
                        format, min_score, min_copy,
                    )
                rows.extend(tmp_rows)
        writer.writerows(rows)
    
    return output_filepath

def annotate_single_segment(
    rgn_dict: dict,
    encoded_seq: np.ndarray,
    job_dir: Path,
    HEADER: list[str],
    format: str,
    min_score: int,
    min_copy: float,
) -> list:
    """
    Annotate a single-segment region (no overlap, no memmap needed).
    Directly writes output rows via the provided csv writer.
    """
    seq_len = len(encoded_seq)
    candidates: list[tuple[int, np.ndarray, str]] = []
    max_score: int = 0
    MAX_MOTIF_LEN: int = max(len(motif) for motif in rgn_dict["motifs"])
    rows: list[list] = []

    # --- refine motifs and compute scores via forward DP ---
    for motif in rgn_dict["motifs"]:
        encoded_motif = encode_seq_to_array(motif)

        # refine motif
        sub_n = seq_len
        sub_m = len(encoded_motif)
        M_ref = np.zeros((sub_n, sub_m), dtype=np.int32)
        I_ref = np.zeros((sub_n, sub_m), dtype=np.int32)
        D_ref = np.zeros((sub_n, sub_m), dtype=np.int32)
        trace = np.zeros((sub_n, sub_m), dtype=np.int8)
        trace_prev_j = np.zeros((sub_n, sub_m), dtype=np.int32)

        profile = _banded_refine_motif(
            encoded_seq, encoded_motif, MAX_PERIOD,
            M_ref, I_ref, D_ref, trace, trace_prev_j,
        )

        encoded_refined = profile.argmax(axis=1)
        encoded_refined = encoded_refined[encoded_refined != 4]

        is_periodic, period = is_periodic_seq(encoded_refined)
        if is_periodic:
            encoded_refined = encoded_refined[:period]

        if len(encoded_refined) == 0:
            continue

        refined_motif = decode_array_to_seq(encoded_refined)

        # forward DP (score only; overlap storage is unnecessary for single segment)
        m = len(encoded_refined)
        M_fwd = np.zeros((seq_len + 1, m), dtype=np.int32)
        I_fwd = np.zeros((seq_len + 1, m), dtype=np.int32)
        D_fwd = np.zeros((seq_len + 1, m), dtype=np.int32)

        pre_M = np.zeros((MAX_MOTIF_LEN, m), dtype=np.int32)
        pre_I = np.zeros((MAX_MOTIF_LEN, m), dtype=np.int32)
        pre_D = np.zeros((MAX_MOTIF_LEN, m), dtype=np.int32)

        banded_dp_align_forward(
            seq=encoded_seq,
            motif=encoded_refined,
            band_width=MAX_PERIOD,
            overlap_length=MAX_MOTIF_LEN,
            pre_M=pre_M,
            pre_I=pre_I,
            pre_D=pre_D,
            is_start=True,
            out_M=M_fwd,
            out_I=I_fwd,
            out_D=D_fwd,
        )

        score = M_fwd[-1, :].max()

        if score >= max_score * SECONDARY_SCORE_RATIO:
            if score > max_score:
                new_threshold = score * SECONDARY_SCORE_RATIO
                candidates = [c for c in candidates if c[0] >= new_threshold]
                max_score = score
            candidates.append((score, encoded_refined, refined_motif))

        del M_ref, I_ref, D_ref, trace, trace_prev_j, M_fwd, I_fwd, D_fwd

    if len(candidates) == 0:
        return []

    if max_score < min_score:
        return []

    candidates = [c for c in candidates if c[0] >= max_score * SECONDARY_SCORE_RATIO]
    candidates.sort(key=lambda x: (-x[0], len(x[1])))

    # --- backward DP + traceback for each candidate ---
    cur_score = 2 ** 31 - 1
    for candidate in candidates:
        score, encoded_motif, motif = candidate
        if score == cur_score:
            continue
        cur_score = score

        m = len(encoded_motif)

        M_bwd = np.zeros((seq_len + 1, m), dtype=np.int32)
        I_bwd = np.zeros((seq_len + 1, m), dtype=np.int32)
        D_bwd = np.zeros((seq_len + 1, m), dtype=np.int32)
        trace_M = np.zeros((seq_len + 1, m), dtype=np.int8)
        trace_I = np.zeros((seq_len + 1, m), dtype=np.int8)
        trace_D = np.zeros((seq_len + 1, m), dtype=np.int8)

        pre_M = np.zeros((MAX_MOTIF_LEN, m), dtype=np.int32)
        pre_I = np.zeros((MAX_MOTIF_LEN, m), dtype=np.int32)
        pre_D = np.zeros((MAX_MOTIF_LEN, m), dtype=np.int32)

        score_array, band_argmax_j = banded_dp_align_backward(
            seq=encoded_seq,
            motif=encoded_motif,
            band_width=MAX_PERIOD,
            overlap_length=MAX_MOTIF_LEN,
            pre_M=pre_M,
            pre_I=pre_I,
            pre_D=pre_D,
            is_start=True,
            trace_M=trace_M,
            trace_I=trace_I,
            trace_D=trace_D,
            out_M=M_bwd,
            out_I=I_bwd,
            out_D=D_bwd,
        )

        best_i_trace = seq_len
        if best_i_trace < 1:
            best_i_trace = 1

        best_state = int(score_array[best_i_trace - 1, :].argmax())
        end_j = int(band_argmax_j[best_i_trace - 1, best_state])

        ops, tb_start_i, tb_start_j = traceback_banded_roll_motif(
            trace_M=trace_M,
            trace_I=trace_I,
            trace_D=trace_D,
            best_i=best_i_trace,
            best_j=end_j,
            m=m,
            seq=encoded_seq,
            motif=encoded_motif,
            stop_i=0,
        )

        # adjust motif based on starting position
        if tb_start_j > 0:
            adjusted_motif = np.concatenate((encoded_motif[tb_start_j:], encoded_motif[:tb_start_j]))
        else:
            adjusted_motif = encoded_motif

        rgn_dict["motif"] = decode_array_to_seq(adjusted_motif)
        rgn_dict["period"] = len(adjusted_motif)
        rgn_dict["consensusSize"] = len(adjusted_motif)
        rgn_dict["score"] = score

        cigar = ops_to_cigar(ops, len(adjusted_motif))
        rgn_dict["cigar"] = cigar if not SKIP_CIGAR else "."
        rgn_dict["copyNumber"] = get_copy_number(cigar, rgn_dict["period"])

        if rgn_dict["copyNumber"] < min_copy:
            continue

        if format in ["trf", "bed"]:
            rgn_dict["percentMatches"], rgn_dict["percentIndels"] = calculate_alignment_metrics(
                ops, len(rgn_dict["sequence"])
            )
            nt_comp, entropy = calculate_nucleotide_composition(rgn_dict["sequence"])
            rgn_dict["A"] = nt_comp["A"]
            rgn_dict["C"] = nt_comp["C"]
            rgn_dict["G"] = nt_comp["G"]
            rgn_dict["T"] = nt_comp["T"]
            rgn_dict["entropy"] = entropy

        rows.append([rgn_dict[k] for k in HEADER])

        del M_bwd, I_bwd, D_bwd, trace_M, trace_I, trace_D, pre_M, pre_I, pre_D

    return rows


def annotate_multiple_segment(
    rgn_dict: dict,
    encoded_seq: np.ndarray,
    subregion: list[tuple[int, int]],
    job_dir: Path,
    HEADER: list[str],
    format: str,
    min_score: int,
    min_copy: float,
) -> list:
    """
    Annotate a multi-segment region (uses memmap-backed pools and overlap chaining).
    Directly writes output rows via the provided csv writer.
    """
    seq_len = len(encoded_seq)
    n_sub = len(subregion)
    candidates: list[tuple[int, np.ndarray, str]] = []
    max_score: int = 0
    MAX_MOTIF_LEN: int = max(len(motif) for motif in rgn_dict["motifs"])
    subregion_pre: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    rows: list[list] = []

    # --- allocate memmap-backed pools ---
    pool_dir = Path(tempfile.mkdtemp(prefix="dp_pool_region_", dir=job_dir))
    pool_dir.mkdir(exist_ok=True)

    pool_shape = (seq_len + 1, MAX_MOTIF_LEN)
    pool_M = np.memmap(str(pool_dir / "dp_M.bin"), dtype=np.int32, mode='w+', shape=pool_shape)
    pool_I = np.memmap(str(pool_dir / "dp_I.bin"), dtype=np.int32, mode='w+', shape=pool_shape)
    pool_D = np.memmap(str(pool_dir / "dp_D.bin"), dtype=np.int32, mode='w+', shape=pool_shape)
    pool_trace = np.memmap(str(pool_dir / "dp_trace.bin"), dtype=np.int8, mode='w+', shape=pool_shape)
    pool_trace2 = np.memmap(str(pool_dir / "dp_trace2.bin"), dtype=np.int8, mode='w+', shape=pool_shape)
    pool_trace3 = np.memmap(str(pool_dir / "dp_trace3.bin"), dtype=np.int8, mode='w+', shape=pool_shape)
    pool_trace_prev_j = np.memmap(
        str(pool_dir / "dp_trace_prev_j.bin"),
        dtype=np.int32, mode='w+', shape=(seq_len, MAX_MOTIF_LEN)
    )

    overlap_counter = 0
    overlap_dir = pool_dir / "overlaps"
    overlap_dir.mkdir(exist_ok=True)

    try:
        # --- refine motifs and forward DP with overlap storage ---
        for motif in rgn_dict["motifs"]:
            logger.debug(f"refining motif {motif} for region {rgn_dict['chrom']}:{rgn_dict['start']}-{rgn_dict['end']}")
            motif_len = len(motif)
            encoded_motif = encode_seq_to_array(motif)

            profile = np.zeros((len(encoded_motif), 5), dtype=np.int32)
            for st, en in subregion:
                sub_n = en - st
                sub_m = len(encoded_motif)
                profile += _banded_refine_motif(
                    encoded_seq[st:en], encoded_motif, MAX_PERIOD,
                    pool_M[:sub_n, :sub_m],
                    pool_I[:sub_n, :sub_m],
                    pool_D[:sub_n, :sub_m],
                    pool_trace[:sub_n, :sub_m],
                    pool_trace_prev_j[:sub_n, :sub_m],
                )

            encoded_refined = profile.argmax(axis=1)
            encoded_refined = encoded_refined[encoded_refined != 4]

            is_periodic, period = is_periodic_seq(encoded_refined)
            if is_periodic:
                encoded_refined = encoded_refined[:period]

            if len(encoded_refined) == 0:
                continue

            refined_motif = decode_array_to_seq(encoded_refined)
            subregion_pre[refined_motif] = []

            pre_M = np.zeros((MAX_MOTIF_LEN, len(encoded_refined)), dtype=np.int32)
            pre_I = np.zeros((MAX_MOTIF_LEN, len(encoded_refined)), dtype=np.int32)
            pre_D = np.zeros((MAX_MOTIF_LEN, len(encoded_refined)), dtype=np.int32)

            for s, e in subregion:
                seg_len = e - s
                sub_m = len(encoded_refined)
                pre_M, pre_I, pre_D = banded_dp_align_forward(
                    seq=encoded_seq[s:e],
                    motif=encoded_refined,
                    band_width=MAX_PERIOD,
                    overlap_length=MAX_MOTIF_LEN,
                    pre_M=pre_M,
                    pre_I=pre_I,
                    pre_D=pre_D,
                    is_start=(s == 0),
                    out_M=pool_M[:seg_len + 1, :sub_m],
                    out_I=pool_I[:seg_len + 1, :sub_m],
                    out_D=pool_D[:seg_len + 1, :sub_m],
                )
                overlap_prefix = str(overlap_dir / f"ol_{overlap_counter}")
                overlap_counter += 1
                m_refined = pre_M.shape[1]
                mmap_M = np.memmap(f"{overlap_prefix}_M.bin", dtype=np.int32, mode='w+', shape=(MAX_MOTIF_LEN, m_refined))
                mmap_I = np.memmap(f"{overlap_prefix}_I.bin", dtype=np.int32, mode='w+', shape=(MAX_MOTIF_LEN, m_refined))
                mmap_D = np.memmap(f"{overlap_prefix}_D.bin", dtype=np.int32, mode='w+', shape=(MAX_MOTIF_LEN, m_refined))
                mmap_M[:] = pre_M[-MAX_MOTIF_LEN:, :]
                mmap_I[:] = pre_I[-MAX_MOTIF_LEN:, :]
                mmap_D[:] = pre_D[-MAX_MOTIF_LEN:, :]
                subregion_pre[refined_motif].append((mmap_M, mmap_I, mmap_D))

            score = pre_M[-1, :].max()

            if score >= max_score * SECONDARY_SCORE_RATIO:
                if score > max_score:
                    new_threshold = score * SECONDARY_SCORE_RATIO
                    removed = [c for c in candidates if c[0] < new_threshold]
                    candidates = [c for c in candidates if c[0] >= new_threshold]
                    for _, _, removed_motif in removed:
                        if removed_motif in subregion_pre:
                            for mmap_M, mmap_I, mmap_D in subregion_pre[removed_motif]:
                                if hasattr(mmap_M, 'filename'):
                                    Path(mmap_M.filename).unlink(missing_ok=True)
                                    Path(mmap_I.filename).unlink(missing_ok=True)
                                    Path(mmap_D.filename).unlink(missing_ok=True)
                            del subregion_pre[removed_motif]
                candidates.append((score, encoded_refined, refined_motif))
                max_score = max(max_score, score)

        if len(candidates) == 0:
            return []

        if max_score < min_score:
            return []

        candidates = [c for c in candidates if c[0] >= max_score * SECONDARY_SCORE_RATIO]
        kept_motifs = {c[2] for c in candidates}
        for key in list(subregion_pre.keys()):
            if key not in kept_motifs:
                for mmap_M, mmap_I, mmap_D in subregion_pre[key]:
                    if hasattr(mmap_M, 'filename'):
                        Path(mmap_M.filename).unlink(missing_ok=True)
                        Path(mmap_I.filename).unlink(missing_ok=True)
                        Path(mmap_D.filename).unlink(missing_ok=True)
                del subregion_pre[key]
        candidates.sort(key=lambda x: (-x[0], len(x[1])))

        # --- backward DP + traceback per candidate ---
        cur_score = 2 ** 31 - 1
        for candidate in candidates:
            score, encoded_motif, motif = candidate
            if score == cur_score:
                continue
            cur_score = score

            ops_list: list[list[str]] = []
            start_i_list: list[int] = []
            start_j_list: list[int] = []

            overlaps = subregion_pre[motif]

            for idx in range(n_sub - 1, -1, -1):
                s, e = subregion[idx]
                pre_M, pre_I, pre_D = overlaps[idx]
                seg = encoded_seq[s:e]

                seg_n = seg.shape[0]
                m = len(encoded_motif)

                trace_M = pool_trace[:seg_n + 1, :m]
                trace_I = pool_trace2[:seg_n + 1, :m]
                trace_D = pool_trace3[:seg_n + 1, :m]

                score_array, band_argmax_j = banded_dp_align_backward(
                    seq=seg,
                    motif=encoded_motif,
                    band_width=MAX_PERIOD,
                    overlap_length=MAX_MOTIF_LEN,
                    pre_M=pre_M,
                    pre_I=pre_I,
                    pre_D=pre_D,
                    is_start=(s == 0),
                    trace_M=trace_M,
                    trace_I=trace_I,
                    trace_D=trace_D,
                    out_M=pool_M[:seg_n + 1, :m],
                    out_I=pool_I[:seg_n + 1, :m],
                    out_D=pool_D[:seg_n + 1, :m],
                )

                best_i_trace = seg_n
                if best_i_trace < 1:
                    best_i_trace = 1
                stop_i = 0 if (s == 0) else MAX_MOTIF_LEN

                best_state = int(score_array[best_i_trace - 1, :].argmax())
                end_j = int(band_argmax_j[best_i_trace - 1, best_state])

                ops, tb_start_i, tb_start_j = traceback_banded_roll_motif(
                    trace_M=trace_M,
                    trace_I=trace_I,
                    trace_D=trace_D,
                    best_i=best_i_trace,
                    best_j=end_j,
                    m=m,
                    seq=seg,
                    motif=encoded_motif,
                    stop_i=stop_i,
                )

                ops_list.append(ops)
                start_i_list.append(tb_start_i)
                start_j_list.append(tb_start_j)

            ops_merged: list[str] = []
            for subops in reversed(ops_list):
                ops_merged.extend(subops)

            start_j = start_j_list[-1]
            if start_j > 0:
                adjusted_motif = np.concatenate((encoded_motif[start_j:], encoded_motif[:start_j]))
            else:
                adjusted_motif = encoded_motif

            rgn_dict["motif"] = decode_array_to_seq(adjusted_motif)
            rgn_dict["period"] = len(adjusted_motif)
            rgn_dict["consensusSize"] = len(adjusted_motif)
            rgn_dict["score"] = score

            cigar = ops_to_cigar(ops_merged, len(adjusted_motif))
            rgn_dict["cigar"] = cigar if not SKIP_CIGAR else "."
            rgn_dict["copyNumber"] = get_copy_number(cigar, rgn_dict["period"])

            if rgn_dict["copyNumber"] < min_copy:
                continue

            if format in ["trf", "bed"]:
                rgn_dict["percentMatches"], rgn_dict["percentIndels"] = calculate_alignment_metrics(
                    ops_merged, len(rgn_dict["sequence"])
                )
                nt_comp, entropy = calculate_nucleotide_composition(rgn_dict["sequence"])
                rgn_dict["A"] = nt_comp["A"]
                rgn_dict["C"] = nt_comp["C"]
                rgn_dict["G"] = nt_comp["G"]
                rgn_dict["T"] = nt_comp["T"]
                rgn_dict["entropy"] = entropy

            rows.append([rgn_dict[k] for k in HEADER])

    finally:
        # --- cleanup ---
        subregion_pre.clear()
        del pool_M, pool_I, pool_D
        del pool_trace, pool_trace2, pool_trace3
        del pool_trace_prev_j
        shutil.rmtree(pool_dir, ignore_errors=True)

    return rows

def encode_ops(ops: list[str]) -> np.ndarray:
    """
    Encode the operations into an integer array
    Inputs:
        ops: list[str], the operations
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    band_argmax_j = np.full((n, 3), -1, np.int16)

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
    
    return (
        score_array,
        band_argmax_j,
        trace_M, trace_I, trace_D,
    )

@numba.njit(cache=True)
def _banded_refine_motif(
    seq: np.ndarray,
    motif: np.ndarray,
    band_width: int,
    out_M: np.ndarray,
    out_I: np.ndarray,
    out_D: np.ndarray,
    out_trace_state: np.ndarray,
    out_trace_prev_j: np.ndarray,
) -> np.ndarray:
    """
    Refine the motif using banded dynamic programming.

    DP matrices (M, I, D) and traceback matrices are provided by the caller
    via pre-allocated arrays (can be np.memmap). This avoids large heap
    allocations that would otherwise be cached by the C allocator and never
    returned to the OS.

    Inputs:
        seq: np.ndarray, sequence
        motif: np.ndarray, motif
        band_width: int, band width
        out_M: np.ndarray, pre-allocated (n, m) int32 matrix for M state
        out_I: np.ndarray, pre-allocated (n, m) int32 matrix for I state
        out_D: np.ndarray, pre-allocated (n, m) int32 matrix for D state
        out_trace_state: np.ndarray, pre-allocated (n, m) int8 traceback state
        out_trace_prev_j: np.ndarray, pre-allocated (n, m) int32 traceback prev_j
    Outputs:
        profile: np.ndarray, motif profile of shape (m, 5)
    """
    n = seq.shape[0]
    m = motif.shape[0]
    NEG_INF = -10**9

    # Initialise caller-provided arrays.
    # np.memmap 'w+' files are zero-initialised, so trace arrays are already
    # correct; we only need to fill the DP matrices with NEG_INF.
    out_M.fill(NEG_INF)
    out_I.fill(NEG_INF)
    out_D.fill(NEG_INF)

    # ---- initialize ----
    for j in range(m):
        out_M[0, j] = 0

    # ---- run-length scaling (optional) ----
    run_len = np.ones(n, dtype=np.int32)
    for i in range(1, n):
        if seq[i] == seq[i-1]:
            run_len[i] = run_len[i-1] + 1
        else:
            run_len[i] = 1

    alpha = 0.5
    min_scale = 0.3

    for i in range(1, n):
        j_center = i % m
        j_start = max(0, j_center - band_width)
        j_end = min(m, j_center + band_width + 1)

        si = seq[i]

        # scale gap penalties
        rl = run_len[i]
        scale = 1.0 / (1.0 + alpha*(rl-1))
        if scale < min_scale:
            scale = min_scale

        gap_open_scaled = int(GAP_OPEN_PENALTY * scale)
        gap_extend_scaled = int(GAP_EXTEND_PENALTY * scale)

        for j in range(j_start, j_end):
            prev_j = m - 1 if j == 0 else j - 1

            # match/mismatch score
            s = MATCH_SCORE if si == motif[j] else -MISMATCH_PENALTY

            # ---- M state ----
            best_prev = out_M[i-1, prev_j]
            state = 0

            if out_I[i-1, prev_j] > best_prev:
                best_prev = out_I[i-1, prev_j]
                state = 1
            if out_D[i-1, prev_j] > best_prev:
                best_prev = out_D[i-1, prev_j]
                state = 2

            out_M[i, j] = best_prev + s
            out_trace_state[i, j] = state
            out_trace_prev_j[i, j] = prev_j

            # ---- I state ---- (gap in motif)
            open_i = out_M[i-1, j] - gap_open_scaled
            ext_i = out_I[i-1, j] - gap_extend_scaled
            if open_i > ext_i:
                out_I[i, j] = open_i
                # prev j stays the same
            else:
                out_I[i, j] = ext_i

            # ---- D state ---- (gap in seq)
            open_d = out_M[i, prev_j] - gap_open_scaled
            ext_d = out_D[i, prev_j] - gap_extend_scaled
            if open_d > ext_d:
                out_D[i, j] = open_d
                # prev j stays prev_j
            else:
                out_D[i, j] = ext_d

    # ---- find best score at last row ----
    last_row = n-1
    all_scores = np.empty(m*3, dtype=np.int32)
    for j in range(m):
        all_scores[j*3+0] = out_M[last_row, j]
        all_scores[j*3+1] = out_I[last_row, j]
        all_scores[j*3+2] = out_D[last_row, j]

    best_idx = np.argmax(all_scores)
    best_state = best_idx % 3
    best_j = best_idx // 3

    # ---- traceback to reconstruct motif profile ----
    profile = np.zeros((m, 5), dtype=np.int32)
    i = last_row
    j = best_j
    state = best_state

    while i >= 0 and j >= 0:
        if state == 0:
            profile[j, seq[i]] += 1
            prev_j = out_trace_prev_j[i, j]
            state = out_trace_state[i, j]
            i -= 1
            j = prev_j
        elif state == 1:
            i -= 1
            # stay same j, state change
            state = 0  # or out_trace_state[i, j] if you want more precise
        else: # D_STATE
            profile[j, 4] += 1
            prev_j = out_trace_prev_j[i, j]
            state = 0  # or out_trace_state[i, j]
            j = prev_j

    return profile

@numba.njit(cache=True)
def banded_dp_align_forward(
    seq: np.ndarray,
    motif: np.ndarray,
    band_width: int,
    overlap_length: int,
    pre_M: np.ndarray,
    pre_I: np.ndarray,
    pre_D: np.ndarray,
    is_start: bool,
    out_M: np.ndarray,
    out_I: np.ndarray,
    out_D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward banded dynamic programming alignment.

    DP matrices are written into caller-provided arrays (can be np.memmap)
    so that the large working memory is file-backed and can be unmapped
    (returned to the OS) after the alignment finishes.

    Inputs:
        seq: np.ndarray, sequence
        motif: np.ndarray, motif
        band_width: int, band width
        overlap_length: int, number of overlap rows to return
        pre_M, pre_I, pre_D: np.ndarray, seed rows from previous segment
        is_start: bool, whether this is the first segment
        out_M, out_I, out_D: np.ndarray, pre-allocated (n+1, m) working matrices
    Outputs:
        new_overlap_M, new_overlap_I, new_overlap_D: np.ndarray,
            copies of the last overlap_length rows for chaining
    """
    # score_array[i, s] = max M/I/D in band at DP row i+1; s in {0,1,2} -> M,I,D.
    # band_argmax_j[i, s] = motif column j (0-based) where that row-state max is attained
    # (first j in band scan order on ties, same as strict > updates).
    n = seq.shape[0]
    m = motif.shape[0]
    NEG_INF = -10 ** 9

    # Initialise caller-provided working matrices.
    out_M.fill(NEG_INF)
    out_I.fill(NEG_INF)
    out_D.fill(NEG_INF)

    if not is_start:
        copy_rows = min(overlap_length, pre_M.shape[0], n + 1)
        pre_offset = pre_M.shape[0] - copy_rows
        for r in range(copy_rows):
            for c in range(m):
                out_M[r, c] = pre_M[pre_offset + r, c]
                out_I[r, c] = pre_I[pre_offset + r, c]
                out_D[r, c] = pre_D[pre_offset + r, c]

    # ---- init ----
    for j in range(m):
        out_M[0, j] = 0

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

    if is_start:
        start = 1
    else:
        start = overlap_length

    for i in range(start, n + 1):

        j_center = (i - 1) % m
        j_start = max(0, j_center - band_width)
        j_end   = min(m, j_center + band_width + 1)

        si = seq[i - 1]

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
            best_prev = out_M[i - 1, prev_j]

            v = out_I[i - 1, prev_j]
            if v > best_prev:
                best_prev = v

            v = out_D[i - 1, prev_j]
            if v > best_prev:
                best_prev = v

            out_M[i, j] = best_prev + s

            # ---- I (gap in motif) ----
            open_i = out_M[i - 1, j] - gap_open_scaled
            ext_i  = out_I[i - 1, j] - gap_extend_scaled

            if open_i > ext_i:
                out_I[i, j] = open_i
            else:
                out_I[i, j] = ext_i

            # ---- D (gap in seq) ----
            open_d = out_M[i, prev_j] - gap_open_scaled
            ext_d  = out_D[i, prev_j] - gap_extend_scaled

            if open_d > ext_d:
                out_D[i, j] = open_d
            else:
                out_D[i, j] = ext_d

    # return copies instead of views so the caller can safely unmap out_* arrays
    new_overlap_M = out_M[n + 1 - overlap_length : n + 1, :].copy()
    new_overlap_I = out_I[n + 1 - overlap_length : n + 1, :].copy()
    new_overlap_D = out_D[n + 1 - overlap_length : n + 1, :].copy()

    return (
        new_overlap_M,
        new_overlap_I,
        new_overlap_D
    )

@numba.njit(cache=True)
def banded_dp_align_backward(
    seq: np.ndarray,
    motif: np.ndarray,
    band_width: int,
    overlap_length: int,
    pre_M: np.ndarray,
    pre_I: np.ndarray,
    pre_D: np.ndarray,
    is_start: bool,
    trace_M: np.ndarray,
    trace_I: np.ndarray,
    trace_D: np.ndarray,
    out_M: np.ndarray,
    out_I: np.ndarray,
    out_D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Banded DP on a segment; same transitions and outputs as banded_dp_align.
    Optionally seeds the first overlap_length rows from forward pass (pre_*).
    Traceback matrices are provided by caller (can be np.memmap).
    DP working matrices are also caller-provided so they can be file-backed.
    """
    n = seq.shape[0]
    m = motif.shape[0]
    NEG_INF = -10 ** 9

    score_array = np.full((n, 3), NEG_INF, np.int32)
    band_argmax_j = np.full((n, 3), -1, np.int16)

    out_M.fill(NEG_INF)
    out_I.fill(NEG_INF)
    out_D.fill(NEG_INF)

    if not is_start:
        copy_rows = min(overlap_length, pre_M.shape[0], n + 1)
        for r in range(copy_rows):
            for c in range(m):
                out_M[r, c] = pre_M[r, c]
                out_I[r, c] = pre_I[r, c]
                out_D[r, c] = pre_D[r, c]

    for j in range(m):
        out_M[0, j] = 0

    run_len = np.ones(n, dtype=np.int32)
    for i in range(1, n):
        if seq[i] == seq[i - 1]:
            run_len[i] = run_len[i - 1] + 1
        else:
            run_len[i] = 1

    alpha = 0.5
    min_scale = 0.3

    if is_start:
        start = 1
    else:
        start = overlap_length
    if start < 1:
        start = 1

    for i in range(start, n + 1):
        j_center = (i - 1) % m
        j_start = max(0, j_center - band_width)
        j_end = min(m, j_center + band_width + 1)

        si = seq[i - 1]
        cur_score_m = NEG_INF
        cur_score_i = NEG_INF
        cur_score_d = NEG_INF
        cur_j_m = -1
        cur_j_i = -1
        cur_j_d = -1

        rl = run_len[i - 1]
        scale = 1.0 / (1.0 + alpha * (rl - 1))
        if scale < min_scale:
            scale = min_scale

        gap_open_scaled = int(GAP_OPEN_PENALTY * scale)
        gap_extend_scaled = int(GAP_EXTEND_PENALTY * scale)

        for j in range(j_start, j_end):
            prev_j = m - 1 if j == 0 else j - 1

            s = MATCH_SCORE if si == motif[j] else -MISMATCH_PENALTY

            best_prev = out_M[i - 1, prev_j]
            state = 0
            v = out_I[i - 1, prev_j]
            if v > best_prev:
                best_prev = v
                state = 1
            v = out_D[i - 1, prev_j]
            if v > best_prev:
                best_prev = v
                state = 2

            out_M[i, j] = best_prev + s
            trace_M[i, j] = state

            open_i = out_M[i - 1, j] - gap_open_scaled
            ext_i = out_I[i - 1, j] - gap_extend_scaled
            if open_i > ext_i:
                out_I[i, j] = open_i
                trace_I[i, j] = 0
            else:
                out_I[i, j] = ext_i
                trace_I[i, j] = 1

            open_d = out_M[i, prev_j] - gap_open_scaled
            ext_d = out_D[i, prev_j] - gap_extend_scaled
            if open_d > ext_d:
                out_D[i, j] = open_d
                trace_D[i, j] = 0
            else:
                out_D[i, j] = ext_d
                trace_D[i, j] = 2

            if out_M[i, j] > cur_score_m:
                cur_score_m = out_M[i, j]
                cur_j_m = j
            if out_I[i, j] > cur_score_i:
                cur_score_i = out_I[i, j]
                cur_j_i = j
            if out_D[i, j] > cur_score_d:
                cur_score_d = out_D[i, j]
                cur_j_d = j

        score_array[i - 1, 0] = cur_score_m
        band_argmax_j[i - 1, 0] = cur_j_m
        score_array[i - 1, 1] = cur_score_i
        band_argmax_j[i - 1, 1] = cur_j_i
        score_array[i - 1, 2] = cur_score_d
        band_argmax_j[i - 1, 2] = cur_j_d

    for i in range(1, start):
        j_center = (i - 1) % m
        j_start = max(0, j_center - band_width)
        j_end = min(m, j_center + band_width + 1)
        cur_score_m = NEG_INF
        cur_score_i = NEG_INF
        cur_score_d = NEG_INF
        cur_j_m = -1
        cur_j_i = -1
        cur_j_d = -1
        for j in range(j_start, j_end):
            if out_M[i, j] > cur_score_m:
                cur_score_m = out_M[i, j]
                cur_j_m = j
            if out_I[i, j] > cur_score_i:
                cur_score_i = out_I[i, j]
                cur_j_i = j
            if out_D[i, j] > cur_score_d:
                cur_score_d = out_D[i, j]
                cur_j_d = j
        score_array[i - 1, 0] = cur_score_m
        band_argmax_j[i - 1, 0] = cur_j_m
        score_array[i - 1, 1] = cur_score_i
        band_argmax_j[i - 1, 1] = cur_j_i
        score_array[i - 1, 2] = cur_score_d
        band_argmax_j[i - 1, 2] = cur_j_d

    return score_array, band_argmax_j

def traceback_banded_roll_motif(
    trace_M: np.ndarray, trace_I: np.ndarray, trace_D: np.ndarray,
    best_i: int, best_j: int,
    m: int,
    seq: np.ndarray,
    motif: np.ndarray,
    stop_i: int = 0,
) -> tuple[list[str], int, int]:
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
        stop_i: int, traceback stops when i reaches stop_i (default 0)
    Outputs:
        ops: list[str], atomic operations: '=', 'X', 'I', 'D', '/'
        start_i: int, starting position in seq (1-based DP index; 0 means start from beginning)
        start_j: int, starting position in motif (0-based index)
    Notes:
        state: 0=M (diagonal), 1=I (gap in motif), 2=D (gap in seq)
    """
    i, j = best_i, best_j
    state = 0

    ops: list[str] = []

    while i > stop_i: # index of seq, 1-based
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
def merge_outputs(job_dir: str, rgn_filepaths: list[str]) -> str:
    """
    Merge sorted separate output files into a single sorted output file.
    Uses streaming merge for memory efficiency.
    Inputs:
        job_dir : str, job directory
        rgn_filepaths : list[str], list of output filepaths (first is linked file)
    Outputs:
        output_filepath : str, path to the merged output file
    """
    csv.field_size_limit(2 * 1024 * 1024 * 1024)
    
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
    Run the scan function

    Parameters
    ----------
        cfg : dict[str, Any], configuration dictionary
    
    Returns
    -------
        None
    
    Generates the following files:
        - <prefix>.log # log file
        - <prefix>.tsv # annotation results
        - <prefix>.web_summary.html # web summary of annotation
        - <JOB_DIR>/windows/ # split windows (temp)
        - <JOB_DIR>/stats/ # smoothness score distribution (temp)
        - <JOB_DIR>/raw_rgns/ # raw regions (temp)
        - <JOB_DIR>/polished_rgns/ # polished regions (temp)
        - <JOB_DIR>/annotated_rgns/ # annotated regions (temp)
    """    
    global JOB_DIR
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
    fasta: Iterator[SeqRecord] = read_fasta(cfg["input"])
    logger.info(f"Finished reading fasta file: {cfg["input"]}")

    # split fasta file into windows
    Path(JOB_DIR + "/windows").mkdir(parents=True, exist_ok=True)
    win_filepaths: list[str] = split_fasta_by_window(fasta, cfg["seq_win_size"], cfg["seq_ovlp_size"], JOB_DIR)
    logger.info(f"Finished splitting windows: n = {len(win_filepaths)} windows created")

    # call non-overlapped tandem repeat regions
    Path(JOB_DIR + "/stats").mkdir(parents=True, exist_ok=True)
    Path(JOB_DIR + "/raw_rgns").mkdir(parents=True, exist_ok=True)
    Path(JOB_DIR + "/polished_rgns").mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, list[int], int, int, int]] = [(win_filepath, cfg["ksize"], cfg["max_period"], cfg["rolling_win_size"], cfg["min_smoothness"]) for win_filepath in win_filepaths]
    try:
        with Pool(processes = cfg["threads"]) as pool:
            results: list[str] = []
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
    results: list[str] = identify_region_across_windows(results, cfg["seq_win_size"], cfg["seq_ovlp_size"])
    logger.info(f"Finished identifying regions across multiple windows")

    # annotate regions
    Path(JOB_DIR + "/annotated_rgns").mkdir(parents=True, exist_ok=True)
    output_filepaths: list[str] = [f"{JOB_DIR}/annotated_rgns/{Path(rgn_path).name}" for rgn_path in results]
    tasks: list[tuple[str, str, str, int]] = [(rgn_path, output_filepath, cfg["format"], cfg["min_score"], cfg["min_copy"]) for rgn_path, output_filepath in zip(results, output_filepaths)]
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

    if cfg["format"] == "bed":
        df: pl.DataFrame = pl.read_csv(final_results_dst, separator="\t", has_header=True)
        df = df.with_columns(
            (pl.col("start") - 1).alias("start"),
            pl.format(
                "{}|{}bp|{}",
                pl.col("motif"),
                pl.col("period"),
                pl.col("copyNumber")
            ).alias("name")
        ).select(["chrom", "start", "end", "name", "pseudoScore", "strand", "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"])
        bed_path: str = final_results_dst.replace(".tsv", ".bed")
        df.write_csv(bed_path, separator="\t", include_header=False)
        logger.info(f"Generated bed file: {bed_path}")

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