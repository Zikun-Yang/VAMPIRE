#! /usr/bin/env python3

# data processing
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
# seqeunce alignment packages
import edlib

from vampire._report_utils import(
    ops_to_cigar,
    get_copy_number,
    calculate_alignment_metrics,
    calculate_nucleotide_composition
)

# type definitions
Region = List[Any] # [win_id, chrom, start, end, ksizes, score, periods] : [int, str, int, int, List[int], float, List[int]]
RegionWithMotifAndSeq = List[Any] # [win_id, chrom, start, end, ksizes, score, periods, seq, motifs] : [int, str, int, int, List[int], float, List[int], str, List[str]]

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
        if logger.isEnabledFor(logging.DEBUG):
            windows = ",".join(f"{i}-{i + seq_win_size}" for i in range(0, len(record.seq), seq_win_size - seq_ovlp_size))
            logger.debug(f"Splitting record: {record.id}, windows: {windows}")
        seq = str(record.seq).upper()
        for i in range(0, len(seq), seq_win_size - seq_ovlp_size):
            window = seq[i : i + seq_win_size]
            if i != 0 and len(window) < seq_ovlp_size:
                break

            output_filepath = f"{job_dir}/windows/window_{window_num + 1}.tsv"
            with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t', lineterminator='\n')
                writer.writerow([window_num + 1, record.id, i, min(i+seq_win_size, len(seq)), window])
            window_filepaths.append(output_filepath)
            window_num += 1
            
    return window_filepaths


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
    win = (win_id, chrom, win_start, win_end, seq)

    # call raw regions
    t1 = time.time() # TODO: TO REMOVE
    raw_rgns: List[Region] = call_raw_rgns(job_dir, win, ksizes, max_dist, score_vision_size, min_smoothness)
    t2 = time.time() # TODO: TO REMOVE
    logger.debug(f"window {win_id}: call_raw_rgns finished, time: {t2 - t1}")
    
    # if no raw regions found, return empty list
    if not raw_rgns:
        return []
    
    # sort by coordinates (start coordinate from small to large, end coordinate from large to small)
    raw_rgns.sort(key=lambda x: (x[2], -x[3]))

    # merge overlapped or contained raw regions, merge k into list, use max score
    merged_rgns: List[Region] = merge_rgns(raw_rgns, include_offset=True)
    logger.debug(f"window {win_id}: merge_rgns finished")

    # return empty list if no merged regions
    if not merged_rgns:
        return []

    # remove concatemer motifs
    merged_rgns = remove_concatemer(merged_rgns, seq)
    logger.debug(f"window {win_id}: remove_concatemer finished")

    # write merged raw regions to file
    output_filepath = f"{job_dir}/raw_rgns/window_{win_id}.tsv"
    with open(output_filepath, 'w', newline='', encoding='utf-8') as fi:
        writer = csv.writer(fi, delimiter='\t', lineterminator='\n')
        writer.writerows(merged_rgns)
    logger.debug(f"window {win_id}: write_raw_rgns finished")

    # polish region borders
    t1 = time.time() # TODO: TO REMOVE
    polished_rgns: List[Region] = polish_rgns(merged_rgns, win)
    t2 = time.time() # TODO: TO REMOVE
    logger.debug(f"window {win_id}: polish_rgns finished, time: {t2 - t1}")

    # sort by coordinates
    polished_rgns.sort(key=lambda x: (x[2], -x[3]))

    # merge overlapped polished regions
    merged_rgns: List[Region] = merge_rgns(polished_rgns, include_offset=False)
    logger.debug(f"window {win_id}: merge_rgns finished")

    # filter polished regions that are too short
    final_rgns: List[Region] = list(filter(lambda x: x[3] - x[2] >= MIN_LEN, merged_rgns))
    logger.debug(f"window {win_id}: filter_rgns finished, total {len(final_rgns)} regions left")

    # remove concatemer motifs
    final_rgns = remove_concatemer(final_rgns, seq)
    logger.debug(f"window {win_id}: polished regions remove_concatemer finished")
    """
    for rgn in final_rgns:
        print(f"final region: {rgn}")
    #""" # TODO

    # get motif and sequence, transform coordinates to 1-based global coordinates
    t1 = time.time() # TODO: TO REMOVE
    rgns_with_motif_and_seq: List[RegionWithMotifAndSeq] = get_motif_and_seq(final_rgns, win)
    t2 = time.time() # TODO: TO REMOVE
    logger.debug(f"window {win_id}: get_motif_and_seq finished, time: {t2 - t1}")

    # write polished regions to file
    output_filepath = f"{job_dir}/polished_rgns/window_{win_id}.tsv"
    with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(rgns_with_motif_and_seq)

    logger.debug(f"window {win_id}: call_region finished")

    return output_filepath

def get_mode(nums: List[int]) -> int|None:
    """
    Get mode from a list of elements using numpy
    Input:
        nums: List[int]
    Output:
        mode: int|None, return None if no mode is found
    """
    match len(nums):
        case 0:
            return None
        case 1:
            return nums[0]
        case _:
            nums = np.asarray(nums, dtype=np.int64)
            vals, counts = np.unique(nums, return_counts=True)
            max_freq = counts.max()
            return int(vals[counts == max_freq][0])

@numba.njit(cache=True)
def get_largest_confident_period_by_k(k: int, alpha: float = 0.1) -> int:
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
def is_low_complexity_kmer(kmer: str) -> bool:
    """
    Heuristics to filter low-complexity / uninformative k-mers.
    Input:
        kmer: str
    Output:
        bool: True if the kmer is low-complexity or uninformative
    Examples considered invalid:
        Homopolymers: AAAAA
        Short-period repeats (including truncated repeats): ATATA, ATATAT, ACACAC, etc.
    """
    k = len(kmer)
    if k == 0:
        return True
    # Homopolymer (e.g. AAAAA)
    if len(set(kmer)) == 1:
        return True

    # Repeats of a short period (including truncated last copy).
    # For example, for k=5, "ATATA" is period=2 truncated repeat of "AT".
    # We treat any period < k that can reconstruct the k-mer as low complexity.
    # k is small (e.g. <= 17 in this pipeline), so O(k^2) here is fine.
    for p in range(1, k):
        rep = (kmer[:p] * ((k + p - 1) // p))[:k]
        if rep == kmer:
            return True

    return False

def find_dominant_kmer(seq: str, ksize: int) -> str | None:
    """
    Find the most frequent (dominant) k-mer in a sequence,
    skipping low-complexity k-mers when applicable.
    Input:
        seq: str
        k: int
    Output:
        dominant_kmer: str | None
    """
    VALID_BASES = {"A", "C", "G", "T"}
    encoded_seq = encode_seq(seq)

    counts = defaultdict(int)
    best_kmer = None
    best_count = 0

    for i in range(len(seq) - ksize + 1):
        window = encoded_seq[i : i + ksize]
        if np.any(window == -1):  # rapidly check for invalid bases
            continue
        kmer = seq[i : i + ksize]

        cnt = counts[kmer] + 1
        counts[kmer] = cnt

        if cnt > best_count and not is_low_complexity_kmer(kmer):
            best_kmer = kmer
            best_count = cnt

    # fallback：if all k-mers are low-complexity, return the most frequent k-mer
    if best_kmer is None and counts:
        best_kmer = max(counts.items(), key=lambda x: x[1])[0]

    return best_kmer

def is_low_complexity_kmer_int(kmer_int: int, k: int, cache: dict) -> bool:
    if kmer_int in cache:
        return cache[kmer_int]
    seen = set()
    tmp = kmer_int
    for _ in range(k):
        seen.add(tmp & 0b11)
        tmp >>= 2
        if len(seen) > 2:
            cache[kmer_int] = False
            return False
    cache[kmer_int] = True
    return True

def find_dominant_kmer_int(encoded_seq: np.ndarray, ksize: int) -> int | None:
    """

    """
    n = len(encoded_seq)
    n_positions = n - ksize + 1
    if n_positions <= 0:
        return None
    
    counts = defaultdict(int)
    best_kmer_int = None
    best_count = 0
    low_complex_cache = {}

    mask = (1 << (2 * ksize)) - 1
    kmer_int = 0
    valid_bases = 0

    for i, b in enumerate(encoded_seq):
        if b == -1:
            valid_bases = 0
            kmer_int = 0
            continue
        kmer_int = ((kmer_int << 2) | int(b)) & mask
        valid_bases += 1
        if valid_bases < ksize:
            continue

        cnt = counts[kmer_int] + 1
        counts[kmer_int] = cnt

        if cnt > best_count and not is_low_complexity_kmer_int(kmer_int, ksize, low_complex_cache):
            best_kmer_int = kmer_int
            best_count = cnt

    # fallback
    if best_kmer_int is None and counts:
        best_kmer_int = max(counts.items(), key=lambda x: x[1])[0]

    if best_kmer_int is None:
        return None
    return best_kmer_int

@numba.njit(cache=True)
def count_kmers(encoded_seq: np.ndarray, ksize: int) -> Tuple[int, int]:
    """
    Count the k-mers in the encoded sequence, skip counting if N or other invalid characters are found
    Input:
        encoded: np.ndarray, the encoded sequence, A=0, C=1, G=2, T=3, N and other invalid characters are encoded as -1
        ksize: int, the length of the k-mer
    Output:
        Tuple[int, int]: the most frequent k-mer and its count, return -1 and 0 if no valid k-mer is found
    """
    n = len(encoded_seq)
    n_positions = n - ksize + 1
    if n_positions <= 0:
        return -1, 0

    table_size = 1 << 20 if ksize > 10 else 1 << (2 * ksize)
    mask = table_size - 1
    counts = np.zeros(table_size, dtype=np.int32)

    current_encoding = np.int64(0)
    bit_mask = (np.int64(1) << (2 * ksize)) - 1

    # preheat the first window
    n_count = 0
    for i in range(ksize):
        val = encoded_seq[i]
        if val == -1:
            n_count += 1
        current_encoding = (current_encoding << 2) | max(val, 0) # if current base is N, add 00 to the encoding

    for i in range(n_positions):
        # skip count if N or other invalid characters are found
        if n_count == 0:
            h = (current_encoding ^ (current_encoding >> 16)) & mask
            counts[h] += 1
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

    max_count = 0
    max_encoding = 0
    for encoding in range(table_size):
        if counts[encoding] > max_count:
            max_count = counts[encoding]
            max_encoding = encoding

    return max_encoding, max_count

def get_most_frequent_kmer(seq: str, k: int) -> str | None:
    """
    Get the most frequent k-mer in the sequence
    Input:
        seq: str
        k: int
    Output:
        most_common_seq: str|None
    """
    if len(seq) < k:
        return None
    if k <= 0:
        encoded_seq = encode_seq(seq)
        kmer_val, _ = count_kmers(encoded_seq, k)
        if kmer_val == -1:
            return None
        kmer = decode_kmer(kmer_val, k)
    else:
        encoded_seq = encode_seq(seq)
        kmer_val = find_dominant_kmer_int(encoded_seq, k)
        if kmer_val is None:
            return None
        kmer = decode_kmer(kmer_val, k)


    return kmer

@numba.njit(cache=True)
def split_regions_by_rolling_median(starts: np.ndarray, ends: np.ndarray, 
                                     rolling_median: np.ndarray, 
                                     max_output_size: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Split regions by rolling_median value changes (numba accelerated)
    Inputs:
        starts, ends: original region start and end indices
        rolling_median: rolling_median array
        max_output_size: maximum size for output arrays (should be >= len(starts) * 2)
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
            # split region at change points
            current_start = start_idx
            for j in range(start_idx, end_idx):
                # check if this is a change point
                if rolling_median[j] != rolling_median[j + 1]:
                    # end of current sub-region at j
                    if count < max_output_size:
                        new_starts[count] = current_start
                        new_ends[count] = j
                        count += 1
                    current_start = j + 1
            
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
def call_raw_rgns(
    job_dir: str, 
    win: Tuple[int, str, int, int, str], 
    ksizes: List[int], 
    max_dist: int, 
    score_vision_size: int, 
    min_smoothness: int
    ) -> List[Region]:
    """
    Call the raw regions for the given sequence
    Input:
        job_dir: str, job directory path
        win: Tuple[int, str, int, int, str]
        ksizes: List[int]
        max_dist: int, maximum distance to call regions
        score_vision_size: int, window length to compute smoothness score
        min_smoothness: int, minimum smoothness score to call regions
    Output:
        raw_rgns: List[Region]
    """
    raw_rgns: List[Region] = []
    win_id, chrom, win_start, win_end, seq = win
    encoded_seq = encode_seq(seq)
    for ksize in ksizes:
        piece_rgns: List[Region] = []
        largest_confident_period = get_largest_confident_period_by_k(ksize)

        # calculate distance
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
        smoothness_score = 1.0 / (robust_cv + 0.01)
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
        starts = np.where(diff == 1)[0]     # switch 0 -> 1, 1-based
        ends = np.where(diff == -1)[0] - 1  # switch 1 -> 0, 1-based
        
        # split regions by rolling_median changes (numba accelerated)
        if len(starts) > 0:
            # estimate max output size: worst case is each region splits into many pieces
            # use len(starts) * 10 as a safe upper bound
            max_output_size = max(len(starts) * 10, 100)
            new_starts_arr, new_ends_arr, count = split_regions_by_rolling_median(
                starts, ends, rolling_median, max_output_size
            )
            if count == max_output_size:
                logger.warning(f"max_output_size is too small, increasing to {max_output_size * 100}")
                max_output_size = max(len(starts) * 100, 100)
                new_starts_arr, new_ends_arr, count = split_regions_by_rolling_median(
                    starts, ends, rolling_median, max_output_size
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
            mode_distance = int(get_mode(region_distances))
            
            piece_rgns.append([
                win_id,
                chrom,
                start_idx,       # relative coordinates
                end_idx,         # relative coordinates
                [ksize],         # list of ksizes
                med_score,
                [mode_distance]  # list of mode distances
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
            if rgn[2] - chaining_width < last_rgn[3] and rgn[6][0] == last_rgn[6][0] and rgn[6][0] >= 5 and last_rgn[6][0] >= 5:  # overlapped and have same period
                merged_start = min(last_rgn[2], rgn[2])
                merged_end = max(last_rgn[3], rgn[3])
                merged_score = max(last_rgn[5], rgn[5])  # keep the max score
                chained_rgns[-1] = [
                    last_rgn[0],
                    last_rgn[1],
                    merged_start,
                    merged_end,
                    last_rgn[4] + rgn[4],  # merge ksize lists
                    merged_score,
                    last_rgn[6] + rgn[6]   # merge period lists though they are same
                ]
            else:
                chained_rgns.append(rgn)

        # filter raw regions with too short length but too long period
        raw_rgns.extend([rgn for rgn in chained_rgns if rgn[3] - rgn[2] > rgn[6][0] / 10.0])

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
        # check overlap or containment
        ###is_motif_len_equal = (get_mode(last_rgn[6]) == get_mode(rgn[6]))
        is_motif_len_equal = bool(set(last_rgn[6]) & set(rgn[6]))
        is_overlapped = (rgn[2] - rgn[6][0] < last_rgn[3]) if include_offset else (rgn[2] < last_rgn[3])
        is_contained = (rgn[2] >= last_rgn[2]) and (rgn[3] <= last_rgn[3])
        if is_contained or (is_overlapped and is_motif_len_equal):  # overlapped and have same period
            merged_start = min(last_rgn[2], rgn[2])
            merged_end = max(last_rgn[3], rgn[3])
            merged_ksize = sorted(list(set(last_rgn[4] + rgn[4])))      # merge ksize lists
            merged_score = max(last_rgn[5], rgn[5])  # keep the max score
            merged_period = sorted(list(set(last_rgn[6] + rgn[6])))     # merge period lists
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
def remove_concatemer(rgns: List[Region], seq: str) -> List[Region]:
    """
    Remove the concatemer motif from the candidates
    Input:
        rgns: List[Region]
        seq: str
    Output:
        rgns: List[Region]
    """
    for idx, rgn in enumerate(rgns):
        period_dedup = sorted(set(rgn[6]), reverse=True)
        if len(period_dedup) <= 1:
            continue
        
        collapsed = {p: p for p in period_dedup}
        # Cache motifs to avoid repeated get_most_frequent_kmer calls
        motif_cache = {}
        included_seq = seq[rgn[2] : rgn[3]] # max(0, rgn[2] - period_dedup[i])
        
        for i in range(len(period_dedup)): # longer motif
            # Early exit if already collapsed
            if collapsed[period_dedup[i]] != period_dedup[i]:
                continue
                
            for j in range(len(period_dedup) - 1, i, -1): # shorter motif
                ### logger.debug(f"-----trying to collapse {period_dedup[i]} and {period_dedup[j]}") # TODO
                ###if period_dedup[j] == 1:
                ###    continue
                if not is_possible_concatemer(period_dedup[i], period_dedup[j]):
                    continue
                
                # Get or compute motifs with caching (use original seq slicing logic)
                if period_dedup[i] not in motif_cache:
                    motif_i = get_most_frequent_kmer(included_seq, period_dedup[i])
                    if motif_i is None:
                        break
                    motif_cache[period_dedup[i]] = motif_i
                else:
                    motif_i = motif_cache[period_dedup[i]]
                
                if period_dedup[j] not in motif_cache:
                    motif_j = get_most_frequent_kmer(included_seq, period_dedup[j])
                    if motif_j is None:
                        continue
                    motif_cache[period_dedup[j]] = motif_j
                else:
                    motif_j = motif_cache[period_dedup[j]]
                
                # Build repeated motif_j
                target_len = 2 * period_dedup[i]
                unit_len = len(motif_j)
                repeat = target_len // unit_len
                remain = target_len % unit_len
                motif_j_rep = motif_j * repeat + motif_j[:remain]
                
                results = edlib.align(motif_i, motif_j_rep, mode="HW", task="distance")
                if results is not None and results["editDistance"] <= 0 * period_dedup[i]: # TODO 0.4 0.2  # now exact same match
                    collapsed[period_dedup[i]] = period_dedup[j]
                    break
                """
                if j == len(period_dedup) - 1 or j == len(period_dedup) - 2:
                    print(f"failed to collapse {period_dedup[i]} to {period_dedup[j]}")
                    print(f"motif_i: {motif_i}, motif_j: {motif_j}, motif_j_rep: {motif_j_rep}")
                    print(f"results: {results}")
                #""" # TODO
        collapsed_period = [p for p in collapsed.keys() if p == collapsed[p]]
        ### print(f"collapsed: {collapsed}") # TODO
        ### print(f"original period: {rgn[6]}") # TODO
        rgns[idx][6] = sorted(list(set(collapsed_period)))
        ### print(f"collapsed period: {rgns[idx][6]}") # TODO
    return rgns

# Step 4: polish borders
def polish_rgns(rgns: List[Region], win: Tuple[int, str, int, int, str]) -> List[Region]:
    """
    Annotate the candidates with more accurate period using edlib
    Input:
        rgns: List[window_id, chrom, start, end, k, score, period]
        win: Tuple[int, str, int, int, str], win_id, chrom, start, end, seq
    Output:
        annotated_candidates: List[window_id, chrom, start, end, k, score, period]
    """
    win_id, chrom, win_start, win_end, seq = win
    seq_len = len(seq)
    max_context_len = 5000   # the maximum length of the context

    ref_motifs: List[str] = []
    t1 = time.time() # TODO: TO REMOVE
    for rgn in rgns:
        best_period = get_mode(rgn[6])
        included_end = rgn[3] + 1
        included_start = max(0, rgn[2] - best_period)
        included_seq = seq[included_start : included_end]
        ref_motif = get_most_frequent_kmer(included_seq, best_period)
        ref_motifs.append(ref_motif)
    
    t2 = time.time() # TODO: TO REMOVE
    logger.debug(f"get motif time: {t2 - t1}")

    idx = 0
    polished_rgns: List[Region] = []
    cur_start, cur_end, cur_period = None, None, []
    t1 = time.time() # TODO: TO REMOVE
    while idx < len(rgns):
        # get basic information
        rgn = rgns[idx]
        next_rgn = rgns[idx + 1] if idx + 1 < len(rgns) else None
        
        best_period = get_mode(cur_period + rgn[6])
        extend_len = min(best_period * 500, 5000) # TODO: * 5 -> * 500
        init_len = best_period * 1

        # polish start
        is_added_period = False
        if cur_start is None:
            ref_motif = ref_motifs[idx]
            if ref_motif is None:
                logger.warning(f"ref_motif is None for start polishing! region {rgn[1]}:{rgn[2]}-{rgn[3]}")
                best_start_offset = 0
            else:
                extend_start = max(0, rgn[2] - extend_len)
                extend_end = rgn[2]
                best_start_offset = extend_border(seq[extend_start : extend_end][::-1], ref_motif[::-1], False)
            cur_start = max(0, rgn[2] - best_start_offset)
            cur_period.extend(rgn[6])
            is_added_period = True
        
        # polish end
        if rgn[3] + 1 <= cur_start + best_period * 100:
            included_seq = seq[cur_start : rgn[3] + 1]
        else:
            included_seq = seq[rgn[3] + 1 - best_period * 100 : rgn[3] + 1] # use nearest 100 periods to get motif (hack for satellites)
        ref_motif = get_most_frequent_kmer(included_seq, best_period)
        if ref_motif is None:
            t1 += 1
            included_seq = seq[cur_start : rgn[3] + 1 + init_len]
            ref_motif = get_most_frequent_kmer(included_seq, best_period)
            if ref_motif is None:
                logger.warning(f"ref_motif is None! included_seq = {included_seq}, best_period = {best_period} for region {rgn[1]}:{rgn[2]}-{rgn[3]}")
        else:
            t2 += 1
        
        # the last region
        if next_rgn is None:
            end_seq = seq[rgn[3] + 1 : min(rgn[3] + 1 + extend_len, seq_len)]
            if ref_motif is None:
                logger.warning(f"ref_motif is None for end polishing (last region)! region {rgn[1]}:{rgn[2]}-{rgn[3]}")
                best_end_offset = 0
            else:
                best_end_offset = extend_border(end_seq, ref_motif, False, 0)
            cur_end = rgn[3] + best_end_offset
            polished_rgns.append([win_id, chrom, cur_start, cur_end, rgn[4], rgn[5], cur_period])
            cur_start, cur_end, cur_period = None, None, []
            idx += 1
            continue
        
        # not the last region
        # consider the coordinates and the motif similarity of the next region
        if next_rgn[2] - rgn[3] <= max_context_len:
            ### logger.debug(f"trying to link the region {rgn[1]}:{rgn[2]}-{rgn[3]} and {next_rgn[1]}:{next_rgn[2]}-{next_rgn[3]}") # TODO
            if ref_motif is None or ref_motifs[idx + 1] is None:
                ###logger.debug(f"ref_motif is None for end polishing! region {rgn[1]}:{rgn[2]}-{rgn[3]}") # TODO
                have_downstream = False
            else:
                short_motif, long_motif = ref_motif, ref_motifs[idx + 1]
                short_len, long_len = len(short_motif), len(long_motif)
                if short_len > long_len:
                    short_motif, long_motif = long_motif, short_motif
                    short_len, long_len = long_len, short_len
                
                short_motif_rep = short_motif * (long_len // short_len)
                short_motif_rep = short_motif_rep + short_motif_rep[:-1]
                result = edlib.align(long_motif, short_motif_rep, mode="HW", task="distance")
                is_highly_similar = (result is not None) and (result["editDistance"] <= 0.40 * long_len) # TODO: how to decide the threshold?
                have_downstream = True if is_highly_similar else False
                ### logger.debug(f"is_highly_similar: {is_highly_similar}, identity: {1 - result['editDistance'] / long_len}, short_motif: {short_motif}, long_motif: {long_motif}") # TODO
        else:
            have_downstream = False
        # extend_end = next_rgn[2] + init_len if have_downstream else rgn[3] + 1 + extend_len
        extend_end = next_rgn[3] if have_downstream else rgn[3] + 1 + extend_len
        
        end_seq = seq[rgn[3] + 1 : min(extend_end, seq_len)]
        
        if ref_motif is None:
            logger.warning(f"ref_motif is None for end polishing! region {rgn[1]}:{rgn[2]}-{rgn[3]}")
            best_end_offset = 0
        else:
            best_end_offset = extend_border(end_seq, ref_motif, have_downstream, 0) # TODO old min score is -(rgn[3]-cur_start)

        # record region
        if best_end_offset + rgn[3] <= next_rgn[2]:
            cur_end = rgn[3] + best_end_offset
            polished_rgns.append([win_id, chrom, cur_start, cur_end, rgn[4], rgn[5], cur_period])
            cur_start, cur_end, cur_period = None, None, []
        else:
            if not is_added_period:
                cur_period.extend(rgn[6]) # TODO old version is: cur_period.extend(next_rgn[6]), i think it is wrong?
        idx += 1
    
    t2 = time.time() # TODO: TO REMOVE
    logger.debug(f"polish time: {t2 - t1}")

    return polished_rgns

def extend_border(seq: str, motif: str, have_downstream: bool, min_score_to_link: int = 0) -> int:
    """
    Extend the border of the motif to the sequence, with the minimum score to link the region
    Input:
        seq: str (target)
        motif: str (query)
        have_downstream: bool, whether this region has downstream region
        min_score_to_link: int, the minimum score to link the region
    Output:
        pos: int, 0-based position
    """
    motif_len = len(motif)
    seq_len = len(seq)

    if seq_len == 0:
        return 0

    if motif_len != 1:
        # create doubled motif that contains all circular shifts
        doubled_motif = motif + motif[:-1]
        
        # use global alignment mode to find the best matching substring in doubled_motif that matches seq
        results = edlib.align(seq[:motif_len], doubled_motif, mode="HW", task="path")
        if results["locations"]:
            match_start = results["locations"][0][0]  # 0-based start position in doubled_motif
            # calculate the circular shift offset
            shift_offset = match_start % motif_len
            # get the best circularly shifted motif
            best_motif = motif[shift_offset:] + motif[:shift_offset]
        else:
            best_motif = motif
        motif = best_motif

    # align
    score, end_score, _, _, _, end_i, end_j = banded_dp_roll_motif(
        seq = encode_seq(seq),
        motif = encode_seq(motif),
        band_width = MAX_PERIOD,
        need_end_score = have_downstream
    )

    # if there is no cost to link 2 regions (no gap), return the end position of the sequence   
    if have_downstream and end_score >= min_score_to_link:
        return seq_len

    return end_i + 1

def get_motif_and_seq(rgns: List[Region], win: Tuple[int, str, int, int, str]) -> List[RegionWithMotifAndSeq]:
    """
    Get the candidate motif and sequence of the regions
    Input:
        rgns: List[Region]
        win: Tuple[int, str, int, int, str], win_id, chrom, win_start, win_end, seq
    Output:
        rgns_with_motif_and_seq: List[RegionWithMotifAndSeq]
    """
    win_id, chrom, win_start, win_end, seq = win
    rgns_with_motif_and_seq: List[RegionWithMotifAndSeq] = []
    for rgn in rgns:
        rgn_seq = seq[rgn[2] : rgn[3] + 1]
        rgn_motifs: List[str] = []
        rgn[4] = list(set(rgn[4])) # remove duplicates
        rgn[6] = list(set(rgn[6])) # remove duplicates
        for motif_len in rgn[6]:
            rgn_motif = get_most_frequent_kmer(rgn_seq, motif_len)
            if rgn_motif is not None:
                rgn_motifs.append(rgn_motif)
        rgns_with_motif_and_seq.append([win_id, chrom, rgn[2] + win_start + 1, rgn[3] + win_start + 1, rgn[4], rgn[5], rgn[6], rgn_seq, rgn_motifs])
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

        if len(rows) == 0:#########
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
                row[4], row[6] = ast.literal_eval(row[4]), ast.literal_eval(row[6])
                cur_st_mid_rgn = row
            elif end >= win_start + seq_win_size - seq_ovlp_size and start < win_start + seq_win_size - seq_ovlp_size:
                row[4], row[6] = ast.literal_eval(row[4]), ast.literal_eval(row[6])
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
        rgn1: RegionWithMotifAndSeq
        rgn2: RegionWithMotifAndSeq
    Output:
        bool
    """
    return rgn1[1] == rgn2[1] and max(rgn1[2], rgn2[2]) <= min(rgn1[3], rgn2[3])

def merge_regions(rgn1: RegionWithMotifAndSeq, rgn2: RegionWithMotifAndSeq) -> RegionWithMotifAndSeq:
    """
    Merge two regions
    Input:
        rgn1: RegionWithMotifAndSeq
        rgn2: RegionWithMotifAndSeq
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
def annotate_regions(task: Tuple[str, str, str, int]) -> str:
    """
    Annotate the regions in one window
    Input:
        polished_rgn_path: str, the path of the polished candidates file
        output_filepath: str, the path of the annotated file
        format: str, the format of the output file, select from ["trf", "verbose"]
        min_score: int, the minimum score threshold for the region
    Output:
        output_filepath: str, the path of the annotated file
    """
    polished_rgn_path, output_filepath, format, min_score = task
    csv.field_size_limit(1024 * 1024 * 1024)  # set as 1 GB to avoid overflow
    with open(polished_rgn_path, 'r', newline='', encoding='utf-8') as fi:
        reader = csv.reader(fi, delimiter='\t')
        rows = list(reader)
    output_rows: List[Any] = []

    match format:
        case "brief":
            header = ["chrom", "start", "end", "period", "copyNumber",
                      "score", "motif"]
        case "trf":
            header = ["chrom", "start", "end", "period", "copyNumber", 
                      "consensusSize", "percentMatches", "percentIndels", "score", "A", "C", "G", "T", "entropy", "motif", "sequence", "cigar"]
        case "bed": # bed12 + extra 12 columns
            header = ["chrom", "start", "end", "motif", "pseudoScore", "strand", "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts",
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
        seq_len = len(rgn_dict["sequence"])
        for motif in rgn_dict["motifs"]:
            score, end_score, trace_M, trace_I, trace_D, end_i, end_j = banded_dp_roll_motif(
                seq = encode_seq(rgn_dict["sequence"]),
                motif = encode_seq(motif),
                band_width = MAX_PERIOD,
                need_end_score = True
            )
            # TODO have to align to end!!!!!
            end_i = seq_len - 1
            if score is None:
                raise RuntimeError(f"Score is None for region {row}")
            if score >= max_score * SECONDARY_SCORE_RATIO:
                candidates.append((score, trace_M, trace_I, trace_D, end_i, end_j, motif))
                max_score = max(max_score, score)

        if len(candidates) == 0:
            ### logger.debug(f"No best motif found for region {row}") # TODO
            continue

        # skip the region if the score is too low
        if max_score < min_score:
            ### print(f"skip region {row} because max_scorev {max_score} < min_score {min_score}")
            continue

        # filter and sort candidates by score
        candidates = [candidate for candidate in candidates if candidate[0] >= max_score * SECONDARY_SCORE_RATIO]
        candidates.sort(key=lambda x: (-x[0], len(x[6])))
        cur_score = 2 ** 31 - 1
        for candidate in candidates:
            score, trace_M, trace_I, trace_D, end_i, end_j, motif = candidate
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
                m = len(motif),
                seq = rgn_dict["sequence"],
                motif = motif,
            )
        
            # Adjust motif based on starting position
            if start_j > 0:
                # Rotate motif so that alignment starts from the beginning
                adjusted_motif = motif[start_j:] + motif[:start_j]
            else:
                adjusted_motif = motif

            # refine motif
            ### consensus = xxxxxxx()
        
            rgn_dict["motif"] = adjusted_motif
            rgn_dict["period"] = len(adjusted_motif)
            rgn_dict["consensusSize"] = len(adjusted_motif)
            rgn_dict["score"] = score

            cigar: str = ops_to_cigar(ops, len(adjusted_motif))
            rgn_dict["cigar"] = cigar if not SKIP_CIGAR else "."
            rgn_dict["copyNumber"] = get_copy_number(cigar, rgn_dict["period"])

            # calculate nucleotide composition and entropy
            if format in ["trf", "bed"]:
                rgn_dict["percentMatches"], rgn_dict["percentIndels"] = calculate_alignment_metrics(ops, len(rgn_dict["sequence"]))
                nt_comp, entropy = calculate_nucleotide_composition(rgn_dict["sequence"])
                rgn_dict["A"], rgn_dict["C"], rgn_dict["G"], rgn_dict["T"] = nt_comp["A"], nt_comp["C"], nt_comp["G"], nt_comp["T"]
                rgn_dict["entropy"] = entropy
        
            # add output row
            output_rows.append([rgn_dict[k] for k in header])

    # wrtie resutls
    with open(output_filepath, 'w', newline='', encoding='utf-8') as fo:
        writer = csv.writer(fo, delimiter='\t', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(output_rows)
    
    return output_filepath

def encode_seq(seq: str) -> np.ndarray:
    """
    Encode a sequence into an integer array, A=0, C=1, G=2, T=3, N and other invalid characters are encoded as -1
    Input:
        seq: str, the sequence to encode
    Output:
        np.ndarray, the encoded sequence
    """
    table = np.full(256, -1, dtype=np.int8)
    table[ord('A')] = 0
    table[ord('C')] = 1
    table[ord('G')] = 2
    table[ord('T')] = 3
    return table[np.frombuffer(seq.encode(), dtype=np.uint8)]

def decode_kmer(val: int, k: int) -> str:
    """
    Decode a k-mer from an integer, only for ACGT bases, N and other invalid characters are not supported
    Input:
        val: int, the integer to decode
        k: int, the length of the k-mer
    Output:
        str, the decoded k-mer
    """
    mapping = "ACGT" # N is not supported
    chars = []
    for _ in range(k):
        chars.append(mapping[val & 3])
        val >>= 2
    return ''.join(reversed(chars))

@numba.njit(cache=True)
def banded_dp_roll_motif(
    seq, motif,
    band_width,
    need_end_score = False
):
    """
    Inputs:
        seq, motif: int8 encoded arrays
        band_width: int, band width
        need_end_score: bool, whether to return the end score, force to align to end if set to True
    Outputs:
        best_score: int, best score
        end_score: int|None, end score if need_end_score is True, otherwise None
        trace_M: np.ndarray, traceback matrix for match / mismatch
        trace_I: np.ndarray, traceback matrix for insertion
        trace_D: np.ndarray, traceback matrix for deletion
        best_i: int, best index in seq
        best_j: int, best index in motif
    Rules:
        state: 0=M, 1=I, 2=D
    """
    n = seq.shape[0]
    m = motif.shape[0]
    NEG_INF = -10 ** 9
    M = np.full((n + 1, m), NEG_INF, np.int32)
    I = np.full((n + 1, m), NEG_INF, np.int32)
    D = np.full((n + 1, m), NEG_INF, np.int32)
    trace_M = np.full((n + 1, m), -1, np.int8)
    trace_I = np.full((n + 1, m), -1, np.int8)
    trace_D = np.full((n + 1, m), -1, np.int8)

    # init
    for j in range(m):
        M[0, j] = 0

    best_score = NEG_INF
    end_score = NEG_INF
    best_i = 0
    best_j = 0

    for i in range(1, n + 1): # index of seq, 1-based
        j_center = (i - 1) % m
        j_start = max(0, j_center - band_width)
        j_end   = min(m, j_center + band_width + 1)
        si = seq[i - 1]
        cur_score = NEG_INF

        for j in range(j_start, j_end): # index of motif, 0-based
            # prev_j
            prev_j = m - 1 if j == 0 else j - 1
            # score
            s = MATCH_SCORE if si == motif[j] else - MISMATCH_PENALTY

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

            # ---- I ----
            open_i = M[i - 1, j] - GAP_OPEN_PENALTY
            ext_i  = I[i - 1, j] - GAP_EXTEND_PENALTY
            if open_i > ext_i:
                I[i, j] = open_i
                trace_I[i, j] = 0
            else:
                I[i, j] = ext_i
                trace_I[i, j] = 1

            # ---- D ----
            open_d = M[i, prev_j] - GAP_OPEN_PENALTY
            ext_d  = D[i, prev_j] - GAP_EXTEND_PENALTY
            if open_d > ext_d:
                D[i, j] = open_d
                trace_D[i, j] = 0
            else:
                D[i, j] = ext_d
                trace_D[i, j] = 2

            # ---- best ----
            if M[i, j] > best_score:
                best_score = M[i, j]
                best_i = i
                best_j = j
            if M[i, j] > cur_score:
                cur_score = M[i, j]

        # early exit if no chance to update best score
        if not need_end_score and (n - i) * MATCH_SCORE + cur_score <= best_score:
            break

    if need_end_score:
        end_score = cur_score

    return (
        best_score,
        end_score if need_end_score else None,
        trace_M, trace_I, trace_D,
        best_i, best_j
    )

def traceback_banded_roll_motif(
    trace_M: np.ndarray, trace_I: np.ndarray, trace_D: np.ndarray,
    best_i: int, best_j: int,
    m: int,
    seq: str,
    motif: str,
) -> Tuple[List[str], int, int]:
    """
    Inputs:
        trace_M: np.ndarray, traceback matrix for match / mismatch
        trace_I: np.ndarray, traceback matrix for insertion
        trace_D: np.ndarray, traceback matrix for deletion
        best_i: int, best index in seq
        best_j: int, best index in motif
        m: int, length of motif
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
    
    if not iterators:
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
                results.append(result)
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
    tasks: List[Tuple[str, str, str, int]] = [(rgn_path, output_filepath, cfg["format"], cfg["min_score"]) for rgn_path, output_filepath in zip(results, output_filepaths)]
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
    if not cfg["skip_report"]:
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
        logger.info(f"Skipping report generation")

    logger.info("Bye.")
    
    # copy log file
    shutil.copy2(f"{JOB_DIR}/log.log", f"{cfg["prefix"]}.log")

    logging.shutdown()
    
    # remove temporary files
    if not cfg["debug"]:
        shutil.rmtree(JOB_DIR)
