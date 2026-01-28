#! /usr/bin/env python3

# argument parsing 
import argparse
# data processing
import math
import numpy as np
import polars as pl
from Bio import SeqRecord, SeqIO
from Bio.Seq import Seq
from collections import Counter
from numpy.lib.stride_tricks import sliding_window_view  # rolling median
# multiprocessing
from tqdm import tqdm
from multiprocessing import Pool
# type hints
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
# basic packages for file operations, logging
import os
import re
import ast
import csv
import time
import logging
import heapq
import shutil
from pathlib import Path
# numba packages for speed optimization
import numba
from numba import jit
# seqeunce alignment packages
import edlib
import parasail

from vampire.stats_utils import ops_to_cigar, get_copy_number, calculate_alignment_metrics, calculate_nucleotide_composition
# debugging
import cProfile

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
        for i in range(0, len(record.seq), seq_win_size - seq_ovlp_size):
            window = str(record.seq[i : i + seq_win_size]).upper()
            if i != 0 and len(window) < seq_ovlp_size:
                ###print(f"Window size is less than overlap size: {len(window)} < {overlap_size}")
                continue

            output_filepath = f"{job_dir}/windows/window_{window_num + 1}.tsv"
            with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t', lineterminator='\n')
                writer.writerow([window_num + 1, record.id, i, min(i+seq_win_size, len(record.seq)), window])
            window_filepaths.append(output_filepath)
            window_num += 1
            
    return window_filepaths


"""
#
# codes for calling repeats
#
"""
# main entry
def call_regions(task: Tuple[str, str, List[int], int, int, int, int, bool]) -> str:
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
            min_len: int, minimum length of the alignment to call regions
    Output:
        result: str, raw region file path, the format is:
            List[window_id, chrom, start, end, k, score, period], start and end are 1-based and include the borders
    """
    job_dir, window_filepath, ksizes, max_dist, score_vision_size, min_smoothness, min_len, composite = task
    # read window information
    csv.field_size_limit(1024 * 1024 * 1024)  # set as 1 GB to avoid overflow
    with open(window_filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            window_id, chrom, win_start, win_end, seq = row
    win_start, win_end = int(win_start), int(win_end)
    win = (window_id, chrom, win_start, win_end, seq)

    # call raw regions
    raw_rgns: List[Region] = call_raw_rgns(job_dir, win, ksizes, max_dist, score_vision_size, min_smoothness)
    
    # if no raw regions found, return empty list
    if not raw_rgns:
        return []
    
    # sort by coordinates (start coordinate from small to large, end coordinate from large to small)
    raw_rgns.sort(key=lambda x: (x[2], -x[3]))

    # merge overlapped or contained raw regions, merge k into list, use max score
    merged_rgns: List[Region] = merge_rgns(raw_rgns, include_offset=True)

    # return empty list if no merged regions
    if not merged_rgns:
        return []

    # remove concatemer motifs
    merged_rgns = remove_concatemer(merged_rgns, seq)

    # write merged raw regions to file
    output_filepath = f"{job_dir}/raw_rgns/window_{window_id}.tsv"
    with open(output_filepath, 'w', newline='', encoding='utf-8') as fi:
        writer = csv.writer(fi, delimiter='\t', lineterminator='\n')
        writer.writerows(merged_rgns)

    # polish region borders
    polished_rgns: List[Region] = polish_rgns(merged_rgns, win, composite)

    # sort by coordinates
    polished_rgns.sort(key=lambda x: (x[2], -x[3]))

    # merge overlapped polished regions
    merged_rgns: List[Region] = merge_rgns(polished_rgns, include_offset=False)

    # filter polished regions that are too short
    final_rgns: List[Region] = list(filter(lambda x: x[3] - x[2] >= min_len, merged_rgns))

    # get motif and sequence, transform coordinates to 1-based global coordinates
    rgns_with_motif_and_seq: List[RegionWithMotifAndSeq] = get_motif_and_seq(final_rgns, win)

    # write polished regions to file
    output_filepath = f"{job_dir}/polished_rgns/window_{window_id}.tsv"
    with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerows(rgns_with_motif_and_seq)

    return output_filepath

def get_mode(nums: List[int]) -> int|None:
    """
    Get mode from a list of elements
    Input:
        nums: List[int]
    Output:
        mode: int|None, return None if no mode is found
    """
    if len(nums) == 0:
        return None
    if len(nums) == 1:
        return nums[0]
    # get mode
    counts = Counter(nums)
    max_freq = max(counts.values())
    candidates = [x for x, c in counts.items() if c == max_freq]
    return min(candidates)

@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
def calculate_distance_ARCHIEVED(seq: str, ksize: int = 17, max_dist: int = 10000) -> np.ndarray:
    """
    Calculate the distance of the given sequence
    Input:
        seq: str
        ksize: int
        max_dist: int, maximum distance to record, use np.nan if exceeding
    Output:
        np.ndarray: the distance array, dtype=np.float32 with np.nan for missing values
    """
    n_positions = len(seq) - ksize + 1
    if n_positions <= 0:
        return np.full(0, np.nan, dtype=np.float32)
    
    distances = np.full(n_positions, np.nan, dtype=np.float32)
    # use a fixed size hash table
    max_kmers = min(n_positions // 5, 4 ** ksize + 10)
    hash_table = np.full(max_kmers, -1, dtype=np.int32)
    
    for i in range(n_positions):
        # compute the simple hash of the kmer
        kmer_hash = 0
        for j in range(ksize):
            kmer_hash = (kmer_hash * 31 + ord(seq[i + j])) % max_kmers
        
        la_pos = hash_table[kmer_hash]
        if la_pos != -1:
            dist = i - la_pos
            if dist < max_dist:
                distances[i] = dist
        hash_table[kmer_hash] = i

    return distances


@numba.jit(nopython=True)
def calculate_distance(seq: str, ksize: int = 17, max_dist: int = 10000) -> np.ndarray:
    """
    Calculate the distance of the given sequence
    Input:
        seq: str
        ksize: int
        max_dist: int, maximum distance to record, use np.nan if exceeding
    Output:
        np.ndarray: the distance array, dtype=np.float32 with np.nan for missing values
    """
    n = len(seq)
    n_positions = n - ksize + 1
    if n_positions <= 0:
        return np.full(0, np.nan, dtype=np.float32)
    
    distances = np.full(n_positions, np.nan, dtype=np.float32)
    table_size = 1 << 20 
    mask = table_size - 1
    
    last_pos_table = np.full(table_size, -1, dtype=np.int32)
    # tag_table stores the complete 64-bit encoding, used to resolve conflicts (only encoding consistent is the same k-mer)
    tag_table = np.full(table_size, -1, dtype=np.int64)

    # DNA 2-bit mapping table (A, C, G, T corresponding to ASCII 65, 67, 71, 84)
    # simple mapping: (base >> 1) & 3 
    current_encoding = np.int64(0)
    bit_mask = (np.int64(1) << (2 * ksize)) - 1

    # preheat the first window
    for i in range(ksize):
        val = (ord(seq[i]) >> 1) & 3
        current_encoding = (current_encoding << 2) | val

    for i in range(n_positions):
        # calculate the hash index of the current encoding (modulo or bitwise operation)
        # use a simple perturbation function to reduce conflicts
        h = (current_encoding ^ (current_encoding >> 16)) & mask
        
        # conflict check
        if tag_table[h] == current_encoding:
            # it is the same k-mer
            la_pos = last_pos_table[h]
            dist = i - la_pos
            if dist < max_dist:
                distances[i] = dist
        
        # update the table
        tag_table[h] = current_encoding
        last_pos_table[h] = i

        # rolling update the encoding (move to the next position)
        if i + ksize < n:
            val = (ord(seq[i + ksize]) >> 1) & 3
            current_encoding = ((current_encoding << 2) | val) & bit_mask

    return distances

@numba.jit(nopython=True)
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


def get_most_frequent_kmer(seq: str, k: int) -> str | None:
    """
    Get the most frequent k-mer in the sequence (optimized version)
    Uses sliding window counting to avoid creating all substrings
    Input:
        seq: str
        k: int
    Output:
        most_common_seq: str|None
    """
    if len(seq) < k:
        return None
    
    kmer_counts = Counter()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        kmer_counts[kmer] += 1
    
    if not kmer_counts:
        return None
    
    # sort by counts
    sorted_kmer_counts = sorted(kmer_counts.items(), key=lambda x: x[1], reverse=True)
    # filter out low-complexity kmers
    for kmer, count in sorted_kmer_counts:
        if k > 1 and k < 50 and is_low_complexity_kmer(kmer):
            continue
        return kmer

    # no valid k-mer found
    return sorted_kmer_counts[0][0]

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
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
    for ksize in ksizes:
        piece_rgns: List[Region] = []
        largest_confident_period = get_largest_confident_period_by_k(ksize)

        # calculate distance
        dist = calculate_distance(seq, ksize, min(largest_confident_period, max_dist))
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
                print(f"Warning: max_output_size is too small, increasing to {max_output_size * 100}")
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

        ###raw_rgns.extend([rgn for rgn in piece_rgns if rgn[3] - rgn[2] > rgn[6][0] / 10.0])

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
        is_motif_len_equal = (get_mode(last_rgn[6]) == get_mode(rgn[6]))
        is_overlapped = (rgn[2] - rgn[6][0] < last_rgn[3]) if include_offset else (rgn[2] < last_rgn[3])
        is_contained = (rgn[2] >= last_rgn[2]) and (rgn[3] <= last_rgn[3])
        if is_contained or (is_overlapped and is_motif_len_equal):  # overlapped and have same period (cancelled)
            merged_start = min(last_rgn[2], rgn[2])
            merged_end = max(last_rgn[3], rgn[3])
            merged_ksize = last_rgn[4] + rgn[4]      # merge ksize lists
            merged_score = max(last_rgn[5], rgn[5])  # keep the max score
            merged_period = last_rgn[6] + rgn[6]     # merge period lists
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
        
        for i in range(len(period_dedup)): # longer motif
            # Early exit if already collapsed
            if collapsed[period_dedup[i]] != period_dedup[i]:
                continue
                
            for j in range(len(period_dedup) - 1, i, -1): # shorter motif
                if period_dedup[j] == 1:
                    continue
                if not is_possible_concatemer(period_dedup[i], period_dedup[j]):
                    continue
                
                # Get or compute motifs with caching (use original seq slicing logic)
                if period_dedup[i] not in motif_cache:
                    motif_i = get_most_frequent_kmer(seq[max(0, rgn[2] - period_dedup[i]) : rgn[3]], period_dedup[i])
                    if motif_i is None:
                        break
                    motif_cache[period_dedup[i]] = motif_i
                else:
                    motif_i = motif_cache[period_dedup[i]]
                
                if period_dedup[j] not in motif_cache:
                    motif_j = get_most_frequent_kmer(seq[max(0, rgn[2] - period_dedup[j]) : rgn[3]], period_dedup[j])
                    if motif_j is None:
                        continue
                    motif_cache[period_dedup[j]] = motif_j
                else:
                    motif_j = motif_cache[period_dedup[j]]
                
                # Build repeated motif_j
                motif_j_rep = motif_j * (period_dedup[i] // period_dedup[j])
                motif_j_rep += motif_j[: (period_dedup[i] % period_dedup[j])]
                
                results = edlib.align(motif_i, motif_j_rep, mode="SHW", task="distance")
                if results is not None and results["editDistance"] <= 0.2 * period_dedup[i]:
                    collapsed[period_dedup[i]] = period_dedup[j]
                    break
        collapsed_period = [collapsed[p] for p in rgn[6]]
        rgns[idx][6] = collapsed_period
    return rgns

# Step 4: polish borders
def polish_rgns(rgns: List[Region], win: Tuple[int, str, int, int, str,], composite: bool) -> List[Region]: # TODO
    """
    Annotate the candidates with more accurate period using edlib
    Input:
        rgns: List[window_id, chrom, start, end, k, score, period]
        win: Tuple[int, str, int, int, str], win_id, chrom, start, end, seq
        composite: bool, whether to include composite tandem repeats
    Output:
        annotated_candidates: List[window_id, chrom, start, end, k, score, period]
    """
    scorescore_list = [] # TODO: TO REMOVE
    win_id, chrom, win_start, win_end, seq = win
    seq_len = len(seq)
    max_context_len = 1000   # the maximum length of the context

    ref_motifs: List[str] = []
    for rgn in rgns:
        best_period = get_mode(rgn[6])
        included_end = rgn[3] + 1
        included_start = max(0, rgn[2] - best_period)
        included_seq = seq[included_start : included_end]
        ref_motif = get_most_frequent_kmer(included_seq, best_period)
        ref_motifs.append(ref_motif)
    
    idx = 0
    polished_rgns: List[Region] = []
    cur_start, cur_end, cur_period = None, None, []
    t1, t2 = 0, 0 # TO REMOVE
    while idx < len(rgns):
        # get basic information
        rgn = rgns[idx]
        next_rgn = rgns[idx + 1] if idx + 1 < len(rgns) else None

        """
        if next_rgn is not None and next_rgn[2] < rgn[3]:
            print(f"next_rgn[2] = {next_rgn[2]}, rgn[3] = {rgn[3]}")
            print(f"next_rgn = {next_rgn}, rgn = {rgn}")
            print(f"next_rgn motif length = {len(ref_motifs[idx + 1])}, rgn motif length = {len(ref_motifs[idx])}")
        """
        
        best_period = get_mode(cur_period + rgn[6])
        extend_len = best_period * 5
        init_len = best_period * 1

        # polish start
        is_added_period = False
        if cur_start is None:
            ref_motif = ref_motifs[idx]
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
                print(f"ref_motif is None! included_seq = {included_seq}, best_period = {best_period} for region {rgn[1]}:{rgn[2]}-{rgn[3]}")
                print(f"cur_start = {cur_start}, rgn[3] = {rgn[3]}")
                print(f"seq = {seq[cur_start : rgn[3] + 1]}")
                print(f"seq_len = {len(seq)}")
                print(f"best_period = {best_period}")
                print(f"init_len = {init_len}")
                print(f"included_seq_len = {len(included_seq)}")
                print(f"included_seq = {included_seq}")
        else:
            t2 += 1
        
        # the last region
        if next_rgn is None:
            end_seq = seq[rgn[3] + 1 : min(rgn[3] + 1 + extend_len, seq_len)]
            best_end_offset = extend_border(end_seq, ref_motif, False, 0)
            cur_end = rgn[3] + best_end_offset
            polished_rgns.append([win_id, chrom, cur_start, cur_end, rgn[4], rgn[5], cur_period])
            cur_start, cur_end, cur_period = None, None, []
            idx += 1
            continue
        
        # not the last region
        if composite:
            # only consider the coordinates of the next region
            have_downstream = True if next_rgn[2] - rgn[3] <= max_context_len else False
            extend_end = next_rgn[2] if have_downstream else rgn[3] + 1 + extend_len
        else:
            # consider the coordinates and the motif similarity of the next region
            if next_rgn[2] - rgn[3] <= max_context_len:
                short_motif, long_motif = ref_motif, ref_motifs[idx + 1]
                short_len, long_len = len(short_motif), len(long_motif)
                if short_len > long_len:
                    short_motif, long_motif = long_motif, short_motif
                    short_len, long_len = long_len, short_len
                
                short_motif_rep = short_motif * (long_len // short_len)
                short_motif_rep = short_motif_rep + short_motif_rep[:-1]
                result = edlib.align(long_motif, short_motif_rep, mode="HW", task="path")
                is_highly_similar = (result is not None) and (result["editDistance"] <= 0.40 * long_len)
                have_downstream = True if is_highly_similar else False
            else:
                have_downstream = False
            # extend_end = next_rgn[2] + init_len if have_downstream else rgn[3] + 1 + extend_len
            extend_end = next_rgn[3] if have_downstream else rgn[3] + 1 + extend_len
        
        end_seq = seq[rgn[3] + 1 : min(extend_end, seq_len)]
        
        best_end_offset = extend_border(end_seq, ref_motif, have_downstream, -(rgn[3]-cur_start))

        if False and cur_start >= 215000 and rgn[3] + best_end_offset <= 221000:
            print(f"start = {cur_start}, end = {rgn[3] + best_end_offset}")
            print(f"long_motif = {long_motif}, short_motif = {short_motif}")
            print(f"result = {result}")
            print(f"is_highly_similar = {is_highly_similar}, have_downstream = {have_downstream}")
            print(f"end_seq_len = {len(end_seq)}, best_end_offset = {best_end_offset}, have_downstream = {have_downstream}")
            print("#################################################")

        # record region
        if best_end_offset + rgn[3] < next_rgn[2]:
            cur_end = rgn[3] + best_end_offset
            polished_rgns.append([win_id, chrom, cur_start, cur_end, rgn[4], rgn[5], cur_period])
            cur_start, cur_end, cur_period = None, None, []
        else:
            if not is_added_period:
                cur_period.extend(rgn[6]) # old version is: cur_period.extend(next_rgn[6]), i think it is wrong?
        idx += 1
    
    """
    print(f"# polished_rgns = {len(polished_rgns)}")
    print(f"# t1 = {t1}, t2 = {t2}")
    print(f"########################################")
    """

    # save the scorescore_list
    """
    with open(f"scorescore_list_{win_id}.txt", "w") as f:
        for score in scorescore_list:
            f.write(f"{score}\n") # TODO: TO REMOVE
    """

    return polished_rgns # TODO: TO REMOVE

def make_semi_global_alignment(query: str, target: str) -> Dict[str, str|int]:
    """
    Semi-global alignment for 3' end extension (free end gaps on right end)
    Input:
        query: str - the query sequence (motif)
        target: str - the target sequence (genomic sequence)
    Output:
        dict containing alignment score, CIGAR string, start_pos, end_pos
        'score': alignment_score,
        'cigar': cigar_string,
        'query_start': query_start, 0-based
        'query_end': query_end, 0-based
        'target_start': target_start, 0-based
        'target_end': target_end, 0-based
    """

    # semi-global alignment with free end gaps on the right end of query and target
    result = parasail.sg_qe_de_trace_striped_sat(
        query,                 # query sequence (motif)
        target,                # target sequence (genomic sequence)
        GAP_OPEN_PENALTY,      # gap open penalty
        GAP_EXTEND_PENALTY,    # gap extend penalty
        GLOBAL_MATRIX          # predefined scoring matrix
    )
    
    alignment_score = result.score
    cigar_string = result.cigar.decode.decode() if result.cigar else None
    
    query_start = result.cigar.beg_query if result.cigar else 0
    target_start = result.cigar.beg_ref if result.cigar else 0
    query_end = result.end_query + 1
    target_end = result.end_ref + 1

    return {
        'score': alignment_score,
        'cigar': cigar_string,
        'query_start': query_start,
        'query_end': query_end,
        'target_start': target_start,
        'target_end': target_end
    }

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

    motif_rep = motif * (seq_len // motif_len + 1)
    results = make_semi_global_alignment(motif_rep, seq)

    # if there is no cost to link 2 regions (no gap), return the end position of the sequence
    logger.debug(f"have_downstream = {have_downstream} score = {results['score']}")
    
    if have_downstream and results["score"] >= min_score_to_link:
        return seq_len

    path = results["cigar"]

    # transform the cigar string to a list of tuples
    cigar_ops = [(int(num), symbol) for num, symbol in re.findall(r'(\d+)([=XIDSM])', path)]

    # curate cigar
    i = 0
    cur_query_pos, cur_target_pos = 0, 0
    while i < len(cigar_ops) - 1:
        is_swap = False
        num, symbol = cigar_ops[i]
        if symbol in ['I','D'] and cigar_ops[i+1][1] in ['=','M']:
            if num == cigar_ops[i+1][0]:
                if symbol == 'I':
                    seq1 = motif_rep[cur_query_pos : cur_query_pos + num]
                    seq2 = motif_rep[cur_query_pos + num : cur_query_pos + num * 2]     
                else:
                    seq1 = seq[cur_target_pos : cur_target_pos + num]
                    seq2 = seq[cur_target_pos + num : cur_target_pos + num * 2]    
                if seq1 == seq2:
                    is_swap = True
            else:
                if symbol == 'I':
                    seq1 = motif_rep[cur_query_pos : cur_query_pos + num]
                    seq2 = motif_rep[cur_query_pos + num : cur_query_pos + num + cigar_ops[i+1][0]]
                else:
                    seq1 = seq[cur_target_pos : cur_target_pos + num]
                    seq2 = seq[cur_target_pos + num : cur_target_pos + num + cigar_ops[i+1][0]]                    
                # have to be polyA/T/C/G
                seq1_dedup = set(seq1)
                seq2_dedup = set(seq2)
                if seq1_dedup == seq2_dedup and len(seq1_dedup) == 1:
                    is_swap = True
        if is_swap:
            if symbol != 'D':
                cur_query_pos += num
            if symbol != 'I':
                cur_target_pos += num
            if cigar_ops[i+1][1] != 'D':
                cur_query_pos += cigar_ops[i+1][0]
            if cigar_ops[i+1][1] != 'I':
                cur_target_pos += cigar_ops[i+1][0]
            cigar_ops[i], cigar_ops[i+1] = cigar_ops[i+1], cigar_ops[i]    # swap the two operations
            i += 2
        else:
            if symbol != 'D':
                cur_query_pos += num
            if symbol != 'I':
                cur_target_pos += num
            i += 1
    
    cur_query_pos, cur_target_pos = 0, 0
    best_pos = 0
    cur_score = 0
    max_score = 0
    par = {'M': MATCH_SCORE, '=': MATCH_SCORE, 'X': - MISMATCH_PENALTY, 'I': - GAP_EXTEND_PENALTY, 'D': - GAP_EXTEND_PENALTY}
    # TODO: if the last few base are mismatch due to mutation, could we add manually?
    # query is motif_rep, target is seq
    for i in range(len(cigar_ops)):
        num, symbol = cigar_ops[i]
        # ignore the mismatch/indel at the head of the sequence (avoid the shift of the reference motif)
        if not (cur_query_pos == 0 and symbol in ['I', 'D']):
            cur_score += num * par[symbol]
        if symbol != 'D':
            cur_query_pos += num
        if symbol != 'I':
            cur_target_pos += num
        if cur_score > max_score:
            max_score = cur_score
            best_pos = cur_query_pos
    return best_pos

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
            "start": row[2],
            "end": row[3],
            "ksizes": row[4].translate(str.maketrans("", "", "[] ")),
            "smoothness": row[5],
            "periods": ast.literal_eval(row[6]),
            "sequence": row[7],
            "motifs": ast.literal_eval(row[8]),
        }
        max_score: int = 0
        best_motif: str = None
        best_trace_M, best_trace_I, best_trace_D, best_end_i, best_end_j = None, None, None, 0, 0
        for motif in rgn_dict["motifs"]:
            score, trace_M, trace_I, trace_D, end_i, end_j = banded_dp_roll_motif(
                seq = encode_seq(rgn_dict["sequence"].encode("ascii")),
                motif = encode_seq(motif.encode("ascii")),
                band_width = 1000 # TODO
            )
            if score is None:
                raise RuntimeError(f"Score is None for region {row}")
            if score > max_score:
                max_score, best_trace_M, best_trace_I, best_trace_D, best_end_i, best_end_j = score, trace_M, trace_I, trace_D, end_i, end_j
                best_motif = motif

        if best_motif is None: # TODO
            ### print(f"No best motif found for region {row}")
            ### print(f"score = {score}, motif = {motif}, end_i = {end_i}, end_j = {end_j}")
            continue # TODO : TO REMOVE

        # skip the region if the score is too low
        if max_score < min_score:
            ### print(f"Score is too low for region {row[:6]}")
            ### print(f"bast_end_i = {best_end_i}, bast_end_j = {best_end_j}")
            ### print(f"score = {max_score}, min_score = {min_score}")
            continue
        
        # get alignment operations and starting positions
        # The alignment may not start from the beginning of the motif or sequence
        # start_i: starting position in seq (1-based DP index, 0 means start from beginning)
        # start_j: starting position in motif (0-based index, 0 means start from beginning)
        ops: List[str]
        start_i: int
        start_j: int
        ops, start_i, start_j = traceback_banded_roll_motif(
            trace_M = best_trace_M,
            trace_I = best_trace_I,
            trace_D = best_trace_D,
            best_i = best_end_i,
            best_j = best_end_j,
            m = len(best_motif),
            seq = rgn_dict["sequence"],
            motif = best_motif,
        )
        
        # Adjust motif based on starting position
        if start_j > 0:
            # Rotate motif so that alignment starts from the beginning
            adjusted_motif = best_motif[start_j:] + best_motif[:start_j]
        else:
            adjusted_motif = best_motif
        
        rgn_dict["motif"] = adjusted_motif
        rgn_dict["period"] = len(adjusted_motif)
        rgn_dict["consensusSize"] = len(adjusted_motif)
        rgn_dict["score"] = max_score

        cigar: str = ops_to_cigar(ops, len(adjusted_motif))
        rgn_dict["cigar"] = cigar
        rgn_dict["copyNumber"] = get_copy_number(cigar, rgn_dict["period"])

        # calculate nucleotide composition and entropy
        if format in ["trf", "verbose"]:
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

@numba.njit(cache=True)
def encode_seq(s):
    out = np.empty(len(s), np.int8)
    for i, c in enumerate(s):
        if c == 65:      # A
            out[i] = 0
        elif c == 67:    # C
            out[i] = 1
        elif c == 71:    # G
            out[i] = 2
        else:            # T
            out[i] = 3
    return out

@numba.njit(cache=True)
def banded_dp_roll_motif(
    seq, motif,
    band_width
):
    """
    Inputs:
        seq, motif: int8 encoded arrays
        band_width: int, band width
    Outputs:
        best_score: int, best score
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
        if (n - i) * MATCH_SCORE + cur_score <= best_score:
            break

    return (
        best_score,
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
def run_scan(args: argparse.Namespace) -> None:
    """
    Run the scan function.
    Inputs:
        args : argparse.Namespace, command line arguments
    Outputs:
        None
    """
    # convert arguments to dictionary
    cfg: dict[str, Any] = args.__dict__

    # Set global variables for alignment functions
    global MATCH_SCORE, MISMATCH_PENALTY, GAP_OPEN_PENALTY, GAP_EXTEND_PENALTY
    MATCH_SCORE = cfg["match_score"]
    MISMATCH_PENALTY = cfg["mismatch_penalty"]
    GAP_OPEN_PENALTY = cfg["gap_open_penalty"]
    GAP_EXTEND_PENALTY = cfg["gap_extend_penalty"]
    min_len: int = cfg["min_score"] / MATCH_SCORE     # TODO
    
    if MATCH_SCORE < 0 or MISMATCH_PENALTY < 0 or GAP_OPEN_PENALTY < 0 or GAP_EXTEND_PENALTY < 0:
        msg = f"Match score, mismatch penalty, gap open penalty, and gap extend penalty must be non-negative integers"
        logger.error(f"ERROR: {msg}")
        raise ValueError(msg)
    
    global GLOBAL_MATRIX
    GLOBAL_MATRIX = parasail.matrix_create("ACGT", MATCH_SCORE, -MISMATCH_PENALTY)
    
    JOB_DIR = ".vampire/" + time.strftime("%Y%m%d_%H%M%S")
    cfg["job_dir"] = JOB_DIR

    START_TIME: float = time.time()
    logger.info(f"{cfg["threads"]} cores are used")

    # make directory for temporary files
    if Path(JOB_DIR).exists():
        os.remove(JOB_DIR)
        logger.warning(f"{JOB_DIR}/ exists, deleted")
    Path(JOB_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created job directory: {JOB_DIR}")
    
    # read input fasta file
    fasta: List[SeqRecord] = list(read_fasta(cfg["fasta"]))
    logger.info(f"Finished reading fasta file: {cfg["fasta"]}")

    # split fasta file into windows
    Path(JOB_DIR + "/windows").mkdir(parents=True, exist_ok=True)
    win_filepaths: List[str] = split_fasta_by_window(fasta, cfg["seq_win_size"], cfg["seq_ovlp_size"], JOB_DIR)
    logger.info(f"Finished splitting windows: n = {len(win_filepaths)} windows created")

    # call non-overlapped tandem repeat regions
    Path(JOB_DIR + "/stats").mkdir(parents=True, exist_ok=True)
    Path(JOB_DIR + "/raw_rgns").mkdir(parents=True, exist_ok=True)
    Path(JOB_DIR + "/polished_rgns").mkdir(parents=True, exist_ok=True)
    tasks: List[Tuple[str, str, List[int], int, int, int, bool]] = [(JOB_DIR, win_filepath, cfg["ksize"], cfg["max_period"], cfg["rolling_win_size"], cfg["min_smoothness"], min_len, cfg["composite"]) for win_filepath in win_filepaths]
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
    results: List[str] = identify_region_across_windows(results, args.seq_win_size, args.seq_ovlp_size)
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
    if not cfg["skipreport"]:
        # import utils
        from vampire.stats_utils import make_stats
        from vampire import make_report
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

    # remove temporary files
    if cfg["debug"]:
        os.remove(JOB_DIR)

    logger.info(f"Bye.")

if __name__ == "__main__":
    scan()