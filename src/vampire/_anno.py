import ast
import csv
import sys
import resource
import shutil
from pathlib import Path
from typing import Any
from importlib.resources import files
import networkx as nx
import time
import logging
from tqdm import tqdm

import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl
import edlib
from pybktree import BKTree
import numpy as np
from Bio import SeqIO   # I/O processing
from Bio.SeqRecord import SeqRecord
from multiprocessing import Pool    # multi-thread

from vampire._utils import(
    encode_seq_to_array,
    decode_array_to_seq,
    encode_array_to_int,
    decode_int_to_array,
)
from vampire._report_utils import (
    ops_to_cigar,
    get_copy_number,
)

logger = logging.getLogger(__name__)

class Decompose:
    def __init__(self, name, sequence, ksize, args_decomp, args_anno):
        self.name = name
        self.sequence = sequence
        self.ksize = ksize
        self.abud_threshold = args_decomp['abud_threshold']
        self.abud_min = args_decomp['abud_min']
        self.min_similarity = args_anno['annotation_min_similarity']

        # results
        self.kmer: dict[int, int] | None = None
        self.dbg: nx.DiGraph | None = None
        self.motif_df: pl.DataFrame | None = None
        self.motif_list: list[str] | None = None
        self.anno_df: pl.DataFrame | None = None

    def count_kmers(self: 'Decompose') -> dict[int, int]:
        """
        count kmers in the sequence
        Input:
            self: Decompose, the Decompose object (use self.sequence, self.ksize)
        Output:
            dict[int, int], the dictionary of kmers and their counts
        """
        # return if computed before
        if self.kmer is not None:
            return self.kmer
        
        # not compute before
        k_len: int = self.ksize + 1
        encoded_seq: np.ndarray = self.sequence
        encoded_seq_len: int = len(encoded_seq)
        kmer_count_int: dict = {}

        if encoded_seq_len >= k_len:
            for i in range(encoded_seq_len - k_len + 1):
                w = encoded_seq[i : i + k_len]
                key = encode_array_to_int(w)
                if key is None:
                    continue
                kmer_count_int[key] = kmer_count_int.get(key, 0) + 1
        
        max_count: int = max(kmer_count_int.values(), default=0)
        min_count: int = max(self.abud_min, max_count * self.abud_threshold)
        self.kmer = {
            k: v
            for k, v in kmer_count_int.items()
            if v >= min_count
        }
        return self.kmer

    def build_graph(self: 'Decompose') -> nx.DiGraph:
        """
        build De Bruijn graph
        Input:
            self: Decompose, the Decompose object (use self.sequence, self.ksize)
            prefix: str, the prefix of the output file
            seq_name: str, the name of the sequence
            pid: int, the process id
        Output:
            nx.DiGraph, the De Bruijn graph
        """
        # return if computed before
        if self.dbg is not None:
            return self.dbg

        # not compute before
        dbg = nx.DiGraph()
        mask = (1 << (2 * self.ksize)) - 1
        for k in self.kmer:
            prefix = k >> 2
            suffix = k & mask
            dbg.add_edge(prefix, suffix, weight = self.kmer[k])

        self.dbg = dbg

        return self.dbg

    def find_motif(self: 'Decompose') -> pl.DataFrame:
        """
        find motifs in the De Bruijn graph
        Input:
            self: Decompose, the Decompose object (use self.dbg)
        Output:
            pl.DataFrame, the dataframe of motifs
        """
        if self.motif_df is not None:
            return self.motif_df

        # not compute before
        cycles: Iterator[list[int]] = nx.simple_cycles(self.dbg)
        motifs: list[list[str]] = []
        for cycle in cycles:
            # get motif
            encoded_node_seq: list[np.ndarray] = []
            for node in cycle:
                encoded_node_seq.append(decode_int_to_array(node, self.ksize))
            encoded_motif: np.ndarray = np.concatenate([seq[(self.ksize - 1):] for seq in encoded_node_seq])
            motif: str = decode_array_to_seq(encoded_motif)
            # calculate the min weight of the loop
            cot = [self.dbg[cycle[i]][cycle[i+1]]['weight'] for i in range(len(cycle) - 1)]
            cot.append(self.dbg[cycle[-1]][cycle[0]]['weight'])  # add the weight of the last edge
            # update
            motifs.append([motif, None, 'UNKNOWN', min(cot), "denovo"])

        motif_df: pl.DataFrame = pl.DataFrame(
            motifs,
            schema=["motif", "ref_seq", "name", "copy_number", "source"],
            orient="row"
        )
        motif_df = motif_df.sort("copy_number", descending=True)

        self.motif_df = motif_df
        self.motif_list = motif_df['motif'].to_list()

        return self.motif_df

    def annotate_with_motif(self: 'Decompose') -> pl.DataFrame:
        """
        annotate the sequence with motifs
        Input:
            self: Decompose, the Decompose object (use self.sequence, self.motif_list, self.min_similarity)
        Output:
            pl.DataFrame, the dataframe of annotations
        """
        if self.anno_df is not None:
            return self.anno_df

        # get motifs and max distances
        motifs: list[str] = self.motif_list
        encoded_motifs: list[np.ndarray] = [encode_seq_to_array(m) for m in motifs]
        motifs_rc_all: list[str] = [rc(motif) for motif in motifs]
        motifs_rc: list[str] = []
        for m_rc in motifs_rc_all:
            encoded_m_rc: np.ndarray = encode_seq_to_array(m_rc)
            is_dup: bool = False
            for encoded_m in encoded_motifs:
                dist: int = calculate_edit_distance_between_motifs(encoded_m_rc, encoded_m)
                if dist == 0:
                    is_dup = True
            if not is_dup:
                motifs_rc.append(m_rc)

        # find matches for plus and minor chains
        seq: str = decode_array_to_seq(self.sequence)
        max_distances: list[int] = [int(len(motif) * (1 - self.min_similarity)) for motif in motifs]
        motif_match_df: pl.DataFrame = find_similar_match(seq, motifs, max_distances)
        motif_match_df = motif_match_df.with_columns(pl.lit("+").alias("orientation"))
        if len(motifs_rc) > 0:
            max_distances: list[int] = [int(len(motif) * (1 - self.min_similarity)) for motif in motifs_rc]
            motif_rc_match_df: pl.DataFrame = find_similar_match(seq, motifs_rc, max_distances)
            motif_rc_match_df = motif_rc_match_df.with_columns(
                pl.col("motif").map_elements(lambda x: rc(x), return_dtype=pl.Utf8).alias("recovered_motif")
            ).with_columns(
                pl.lit("-").alias("orientation"),
                pl.col("recovered_motif").alias("motif")
            ).drop("recovered_motif")
        else:
            motif_rc_match_df = pl.DataFrame(schema = ["start", "end", "distance", "score", "cigar", "motif", "orientation"])
        result = pl.concat([motif_match_df, motif_rc_match_df]) # ['start', 'end', 'distance', 'score', 'cigar', 'motif', 'orientation'], coordinates are 0-based, end is inclusive

        result = result.sort("start")

        self.anno_df = result

        return self.anno_df

def canonicalize_motif_str(s: str) -> str:
    """
    Return the lexicographically smallest cyclic rotation of a string (Booth algorithm).
    Input:
        s: str
    Output:
        str
    """
    n = len(s)
    if n == 0:
        return s

    i, j, k = 0, 1, 0

    while i < n and j < n and k < n:
        a = s[(i + k) % n]
        b = s[(j + k) % n]

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
    return s[start:] + s[:start]

def rc(seq: str) -> str:
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N':'N'}
    return ''.join(complement[base] for base in reversed(seq))

# Step 1: process N and other invalid characters, split into windows
def preprocess_sequence(
    record: SeqIO.SeqRecord
) -> dict[str, Any]:
    """
    N-mask handling + sliding-window decomposition; returns data needed for global motif merge and annotation.
    Input:
        record: SeqIO.SeqRecord
    Output:
        dict[str, Any]
    """
    seq_name = record.name
    logger.debug(f"Preprocessing {seq_name}")

    seq: str = str(record.seq.upper())
    encoded_seq: np.ndarray = encode_seq_to_array(seq)
    seq_len_with_N: int = len(seq)
    SEQ_STEP_SIZE: int = SEQ_WIN_SIZE - SEQ_OVLP_SIZE

    invalid_mask: np.ndarray = encoded_seq == -1
    valid2raw: np.ndarray = np.where(~invalid_mask)[0]
    raw2valid: np.ndarray = np.full(seq_len_with_N, -1, dtype=np.int32)
    raw2valid[valid2raw] = np.arange(len(valid2raw))

    if invalid_mask.any():
        invalid_pos: list[int] = np.where(invalid_mask)[0].tolist()
        invalid_char: list[str] = [seq[i] for i in invalid_pos]
        invalid_char = list(set(invalid_char))
        logger.warning(f"Invalid characters found in sequence {seq_name}: {invalid_char}")

        # mask invalid char
        encoded_seq: np.ndarray = encoded_seq[encoded_seq != -1]
    else:
        ### empty
        pass
    encoded_seq_len: int = len(encoded_seq)

    idx: int = 0
    task_num: int = max(1, (encoded_seq_len - SEQ_WIN_SIZE) // SEQ_STEP_SIZE + 1)
    tasks: list[tuple[str, int, int, np.ndarray]] = []
    while True:
        start: int = idx * SEQ_STEP_SIZE
        window_seq: np.ndarray = encoded_seq[start : start + SEQ_WIN_SIZE]
        window_seq_len: int = len(window_seq)
        tasks.append(
            (seq_name, idx + 1, task_num, window_seq)
        )
        if window_seq_len < SEQ_WIN_SIZE:
            break
        idx += 1

    return {
        "seq_name": seq_name,
        "have_invalid": invalid_mask.any(),
        "coordinates_raw2valid": raw2valid,
        "coordinates_valid2raw": valid2raw,
        "tasks": tasks
    }

# Step 2: get all motifs
def decompose_sequence(
    task: tuple[str, int, int, np.ndarray]
) -> 'Decompose':
    """
    Decompose sequence into motifs
    Input:
        task: tuple[str, int, int, np.ndarray]
    Output:
        dict[str, Any]
    """
    seq_name, pid, total_task, sequence = task

    decomp_opts = {
        "abud_threshold": CFG["abud_threshold"],
        "abud_min": CFG["abud_min"]
    }
    anno_opts = {"annotation_min_similarity": CFG["annotation_min_similarity"]}
    seg = Decompose(
        name = f"{seq_name}_{pid}",
        sequence = sequence, 
        ksize = CFG["ksize"], 
        args_decomp = decomp_opts, 
        args_anno = anno_opts
    )
    seg.count_kmers()
    seg.build_graph()
    seg.find_motif()
    logger.debug(f"Finished decomposition {seq_name} {pid}/{total_task}")

    return seg

# Step 3: polish motifs by BK-tree search and cancatemer trimming
def polish_motif(
    motif_catalog: pl.DataFrame,
    cfg: dict[str, Any],
) -> pl.DataFrame:
    """
    Polish motifs by BK-tree search and dimer cut; returns polished motif table for one sequence.
    
    Optimizations:
    1. Only search using canonical form (not all rotations) to reduce BK-tree queries
    2. Cache dimer cut results to avoid redundant computation
    3. Use polars vectorized operations where possible
    
    Input:
        motif_catalog: pl.DataFrame with columns [motif, ref_seq, name, copy_number, source],
            motif: the sequence to use in annotation
            ref_seq: the matched reference sequence from database
            name: the matched labal/seq name from database
            copy_number: the estimated copy number from decomposition module
            source: the source of this motif
        cfg: configuration dictionary with finding_min_similarity
    Output:
        pl.DataFrame with updated ref_seq and name
    """
    # separate database and denovo motifs
    motif_catalog_database: pl.DataFrame = motif_catalog.filter(pl.col("source") == "database")
    motif_catalog_denovo: pl.DataFrame = motif_catalog.filter(pl.col("source") == "denovo")
    
    if motif_catalog_denovo.is_empty():
        return motif_catalog
    
    # build BK-tree from database motifs (using canonical forms)
    tree = BKTree(calculate_rotation_invariant_distance)
    ref_to_idx: dict[str, int] = {}
    for idx in range(len(motif_catalog_database)):
        ref_motif = motif_catalog_database.item(row = idx, column = "motif")
        tree.add(ref_motif)
        ref_to_idx[ref_motif] = idx
    
    logger.info(f"Initialized BK-tree with {len(motif_catalog_database)} database motifs")
    
    # prepare results
    polished_rows: list[dict[str, Any]] = []
    
    # sort by copy_number (descending) to process high-confidence motifs first
    motif_catalog_denovo = motif_catalog_denovo.sort("copy_number", descending=True)
    
    for idx in range(len(motif_catalog_denovo)):
        row = motif_catalog_denovo.row(idx, named = True)
        motif: str = row["motif"]
        motif_len: int = len(motif)
        canonical_form: str = row["canonical_motif"]
        finding_min_similarity: float = cfg.get("finding_min_similarity", 0.8)
        max_distance: int = int((1 - finding_min_similarity) * len(motif))
        
        min_distance: int = 1_000_000
        best_motif: str = canonical_form
        best_ref: str = canonical_form
        name: str = "UNKNOWN"
        is_matched: bool = False

        logger.debug(f"Polishing motif {idx}: {motif}")
        
        # search canonical form
        matches = tree.find(motif, max_distance)
        logger.debug(f"Trying match canonical form ...")
        for dist, ref_motif in matches:
            if dist < min_distance:
                min_distance = dist
                ref_row = motif_catalog_database.row(ref_to_idx[ref_motif], named = True)
                best_ref = ref_motif
                phase = calculate_phase_difference(best_ref, motif)
                best_motif = motif[phase:] + motif[: phase]
                name = ref_row["name"]
                is_matched = True
                logger.debug(f"### Succesfully matched!\n")
        
        # also check reverse complement canonical form
        if not is_matched:
            logger.debug(f"Trying match reverse complementary form ...")
            motif_rc = rc(motif)
            matches_rc = tree.find(motif_rc, max_distance)
            for dist, ref_motif in matches_rc:
                if dist < min_distance:
                    min_distance = dist
                    ref_row = motif_catalog_database.row(ref_to_idx[ref_motif], named = True)
                    best_ref = ref_motif
                    phase = calculate_phase_difference(best_ref, motif_rc)
                    best_motif = motif_rc[phase:] + motif_rc[: phase]
                    name = ref_row["name"]
                    is_matched = True
                    logger.debug(f"### Succesfully matched reverse complementary form!\n")
        
        # concatemer processing - only process if not already matched
        is_cut: bool = False
        if not is_matched:
            logger.debug(f"Trying cut concatemer ...")
            best_motif, is_cut = _try_dimer_cut( # TODO TODO TODO
                motif = canonical_form, 
                known_motifs = list(ref_to_idx.keys()), 
                max_ratio=3.0, 
                ratio_tolerance=0.2
            )
            if is_cut:
                logger.debug(f"Cut motif {motif} to {best_motif}")
                best_ref = best_motif
                # add cut motif to tree for subsequent processing
                if best_motif not in ref_to_idx:
                    tree.add(best_motif)
                    ref_to_idx[best_motif] = len(ref_to_idx)
                    new_row = {
                        "motif": best_motif,
                        "ref_seq": best_ref,
                        "name": name,
                        "copy_number": row["copy_number"],
                        "source": "from-cut",
                        "canonical_motif": best_motif
                    }
                    new_row_df = pl.DataFrame([new_row])
                    motif_catalog_database = motif_catalog_database.vstack(new_row_df)
                    logger.debug(f"### Succesfully cut!")
        
        # if still no match, use canonical form and add to tree
        if not is_matched and not is_cut:
            if motif not in ref_to_idx:
                canonical_form = canonicalize_motif_str(motif)
                logger.debug(f"### Motif {idx} is not registered, pushing it into the tree\n")
                tree.add(motif)
                ref_to_idx[motif] = len(ref_to_idx)
                new_row = {
                    "motif": canonical_form,
                    "ref_seq": canonical_form,
                    "name": "UNKNOWN",
                    "copy_number": row["copy_number"],
                    "source": "from-denovo",
                    "canonical_motif": canonical_form
                }
                new_row_df = pl.DataFrame([new_row])
                motif_catalog_database = motif_catalog_database.vstack(new_row_df)
        else:
            # update row with polished motif
            polished_rows.append({
                "motif": best_motif,       # use polished canonical form
                "ref_seq": best_ref,       # use matched reference sequence if found, otherwise same as motif
                "name": name,
                "copy_number": row["copy_number"],
                "source": row["source"],
                "canonical_motif": canonical_form
            })
    
    # Combine database and polished denovo motifs
    if polished_rows:
        polished_denovo_df = pl.DataFrame(polished_rows)
        motif_catalog_database = motif_catalog_database.filter(pl.col("source") == "database")
        result_catalog = pl.concat([motif_catalog_database, polished_denovo_df])
    else:
        result_catalog = motif_catalog_database
    
    return result_catalog

def _try_dimer_cut(
    motif: str,
    known_motifs: list[str],
    max_ratio: float = 3.0,
    ratio_tolerance: float = 0.2
) -> tuple[str, bool]:
    """
    Try to cut a motif into smaller repeating units.
    
    Args:
        motif: The canonical form of the motif to cut
        known_motifs: list of known motifs
        max_ratio: Maximum length ratio to consider for cutting
        ratio_tolerance: Tolerance for ratio being close to integer
    
    Returns:
        tuple of (cut_motif, was_cut)
    """
    original_len = len(motif)
    current_motif = motif
    
    # sort by length (ascending) to try shorter motifs first
    known_motifs.sort(key=len)
    
    while True:
        was_cut_this_round = False
        
        for ref_motif in known_motifs:
            ref_len = len(ref_motif)
            current_len = len(current_motif)
            
            # skip invalid candidates
            if ref_len == 1 or ref_len >= current_len:
                continue
            
            ratio = current_len / ref_len
            if ratio > max_ratio:
                continue
            
            # check if ratio is close to an integer
            nearest_int = round(ratio)
            min_ratio_diff = min(
                abs(ratio - nearest_int),
                abs(ratio - nearest_int - 1)
            )
            if min_ratio_diff >= ratio_tolerance:
                continue
            
            # check if motif starts with the reference motif
            if current_motif.startswith(ref_motif):
                # use edlib to check if cutting improves the match
                before = edlib.align(current_motif, ref_motif, mode="NW")["editDistance"]
                after = edlib.align(current_motif[ref_len:], ref_motif, mode="NW")["editDistance"]
                
                if after < before:
                    logger.debug(
                        f"Dimer cut: {current_motif} -> {current_motif[ref_len:]} | distance: {before} -> {after}"
                    )
                    current_motif = current_motif[ref_len:]
                    current_motif = canonicalize_motif_str(current_motif)
                    was_cut_this_round = True
                    break
        
        if not was_cut_this_round:
            break
    
    was_cut = len(current_motif) < original_len
    return current_motif, was_cut

def annotate_sequence(
    seg: Decompose,
) -> Decompose:
    """
    Annotate sequence with motifs
    Input:
        seg: Decompose
    Output:
        seg: Decompose
    """
    seg.annotate_with_motif()

    anno_df: pl.DataFrame = seg.anno_df
    pid: int = int(seg.name.split("_")[-1])
    anno_df = anno_df.with_columns(
        (pl.col("start") + (pid - 1) * (SEQ_WIN_SIZE - SEQ_OVLP_SIZE)).alias("start"),
        (pl.col("end") + (pid - 1) * (SEQ_WIN_SIZE - SEQ_OVLP_SIZE)).alias("end")
    )
    anno_df = anno_df.sort("end")
    logger.debug(f"Finished annotation: {seg.name}")
    return seg

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

def traceback_banded_roll_motif(
    trace_M: np.ndarray, trace_I: np.ndarray, trace_D: np.ndarray,
    best_i: int, best_j: int,
    m: int,
    seq: np.ndarray,
    motif: np.ndarray,
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

def find_similar_match(seq: str, motifs: list[str], max_distances: list[int]) -> pl.DataFrame:
    """
    Find similar matches of motifs in the sequence.
    Input:
        seq: str, the input sequence
        motifs: list[str], the list of motifs to match
        max_distances: list, the list of maximum edit distances for each motif
    Output:
        motif_match_df: pl.DataFrame, columns ['start', 'end', 'distance', ''score', 'cigar', 'motif'], coordinates are 0-based, end is inclusive
    """
    motif_pos_rows: list[dict] = []
    seq_len: int = len(seq)

    for motif_id, motif in enumerate(motifs):
        encoded_motif: np.ndarray = encode_seq_to_array(motif)
        motif_len: int = len(motif)
        max_distance = max_distances[motif_id]
        score_array, band_argmax_j, trace_M, trace_I, trace_D = banded_dp_align(
            seq = encode_seq_to_array(seq),
            motif = encoded_motif,
            band_width = motif_len,  # full band
            align_to_end = True
        )

        state: int= np.argmax(score_array[seq_len - 1, :]) # 0 -> M, 1 -> I, 2 -> D
        best_j: int = band_argmax_j[seq_len - 1, state]

        # calculate edit distance from traceback
        ops, _, _ = traceback_banded_roll_motif(
            trace_M = trace_M,
            trace_I = trace_I,
            trace_D = trace_D,
            best_i = seq_len,
            best_j = best_j,
            m = motif_len,
            seq = seq,
            motif = motif
        )
        rows: list[tuple[int, int, int, int, str]] = split_ops(ops, len(motif), max_distance)
        motif_pos: pl.DataFrame = pl.DataFrame(
            rows,
            schema=["start", "end", "distance", "score", "cigar"],
            orient="row"
        )
        motif_pos = motif_pos.with_columns(pl.lit(motif).alias("motif"))
        motif_pos_rows.extend(motif_pos.to_dicts())

    if not motif_pos_rows:
        return pl.DataFrame(
            schema=["start", "end", "distance", "score", "cigar", "motif"]
        )

    # only keep the rows with min distance among rows with the same start and end
    motif_match_df: pl.DataFrame = pl.DataFrame(motif_pos_rows) 

    min_dist_df = motif_match_df.group_by(["start", "end"]).agg(
        pl.col("distance").min().alias("min_distance")
    )
    motif_match_df = motif_match_df.join(min_dist_df, on=["start", "end"])
    motif_match_df = motif_match_df.filter(pl.col("distance") == pl.col("min_distance"))
    motif_match_df = motif_match_df.drop("min_distance").sort("start")

    motif_match_df = (motif_match_df
        .group_by(["start", "end"], maintain_order=True)
        .agg([
            pl.col("distance").min().alias("distance"),
            pl.col("score").first().alias("score"),
            pl.col("cigar").first().alias("cigar"),
            pl.col("motif").first().alias("motif")
        ])
        .sort("start")
    )

    return motif_match_df

def split_ops(
    ops: list[str], 
    m: int,
    max_distance: int
) -> list[tuple[int, int, int, int, str]]:
    """
    Split the operations into blocks
    Inputs:
        ops : list[str], atomic operations: '=', 'X', 'I', 'D'
        m : int, length of the motif
    Outputs:
        list[tuple[int, int, int, int, str]], list of blocks: (start, end, distance, score, cigar), coordinates are 0-based, end is inclusive
    """
    cur_phase: int = -1 # current phase in the motif, from 0 to m-1
    cur_pos: int = -1 # current position on sequence, 0-based
    cur_score: int = 0
    start: int = 0
    ops_start: int = 0
    distance: int = 0
    split_results: list[tuple[int, int, int, int, str]] = []
    for i, op in enumerate(ops):
        match op:
            case "=":
                cur_score += MATCH_SCORE
                cur_pos += 1
                cur_phase += 1
            case "X":
                cur_score -= MISMATCH_PENALTY
                cur_pos += 1
                cur_phase += 1
                distance += 1
            case "I":
                cur_score -= GAP_EXTEND_PENALTY
                cur_pos += 1
                distance += 1
            case "D":
                cur_score -= GAP_EXTEND_PENALTY
                cur_phase += 1
                distance += 1
        
        if cur_phase == m - 1:
            if distance <= max_distance:
                split_results.append((start, cur_pos, distance, cur_score, ops_to_cigar(ops[ops_start: i + 1], m))) # 0-based coordinates; end is inclusive
            start = cur_pos + 1
            ops_start = i + 1
            distance = 0
            cur_phase = -1
            cur_score = 0
    
    if cur_phase > 0:
        split_results.append((start, cur_pos, distance, cur_score, ops_to_cigar(ops[ops_start:], m)))

    return split_results

# Step 4: dynamic programming
def run_dp(
    task: tuple[str, str, pl.DataFrame],
) -> pl.DataFrame:
    """
    run dynamic programming to find the best combination of motif matches for one sequence
    Inputs:
        task: tuple[str, str, pl.DataFrame], (seq_name, seq, anno_df)
    Outputs:
        pl.DataFrame: the best combination of motif matches
    """
    seq_name, seq, anno_df = task
    seq_len: int = len(seq)

    # dynamic programming to find the best combination of motif matches
    dp: np.ndarray = np.zeros(seq_len + 1, dtype=np.ndarray) # dp[x] means the best score after annotate the first x bases
    pre: np.ndarray = np.full(seq_len + 1, -1)

    aln_idx: int = 0
    for seq_idx in range(1, seq_len + 1): # here is 1-based, need to transform to 0-based when compare with anno_df
        # --- skip ---
        if pre[seq_idx - 1] != -1:
            dp[seq_idx] = dp[seq_idx - 1] - GAP_OPEN_PENALTY
        else:
            dp[seq_idx] = dp[seq_idx - 1] - GAP_EXTEND_PENALTY
        
        # --- match ---
        while aln_idx < anno_df.shape[0]:
            cur_end: int = anno_df.item(row = aln_idx, column = "end")
            if cur_end > seq_idx - 1:
                break
            if cur_end < seq_idx - 1:
                aln_idx += 1
                continue

            start: int = anno_df.item(row = aln_idx, column = "start")
            score: int = anno_df.item(row = aln_idx, column = "score") + dp[start]

            if score > dp[seq_idx]:
                dp[seq_idx] = score
                pre[seq_idx] = aln_idx
            aln_idx += 1

    # retrace the best path
    rows: list[dict] = []
    seq_idx: int = seq_len
    while seq_idx > 0:
        aln_idx: int = pre[seq_idx]
        if aln_idx == -1:
            seq_idx -= 1
            continue

        row: dict = anno_df.row(aln_idx, named = True)
        row["sequence"] = seq[row["start"]: row["end"] + 1]
        rows.append(row)
        seq_idx = row["start"]
        
    row_len: int = len(rows) 
    result_df: pl.DataFrame = pl.DataFrame(rows, schema=["start", "end", "distance", "score", "cigar", "motif", "orientation", "sequence"], orient="row").sort("start")

    # add segments that are not annotated with motifs
    if result_df.shape[0] == 0:
        rows.append(
            {
                "start": 0,
                "end": seq_len - 1,
                "distance": None,
                "score": GAP_OPEN_PENALTY + (seq_len - 1) * GAP_EXTEND_PENALTY,
                "cigar": f"{seq_len}N",
                "motif": None,
                "orientation": None,
                "sequence": seq
            }
        )
    else:
        for idx in range(result_df.shape[0]):
            start = result_df["end"][idx]
            if idx == result_df.shape[0] - 1:
                end = seq_len
            else:
                end = result_df["start"][idx + 1] 
            if start + 1 <= end - 1:
                rows.append(
                    {
                        "start": start + 1,
                        "end": end - 1,
                        "distance": None,
                        "score": GAP_OPEN_PENALTY + (end - start - 2) * GAP_EXTEND_PENALTY,
                        "cigar": f"{end - start - 1}N",
                        "motif": None,
                        "orientation": None,
                        "sequence": seq[start + 1: end] # here is not end - 1 because the end is the inclusive start of next annotated region
                    }
                )
    if len(rows) > row_len:
        result_df = pl.DataFrame(rows).sort("start")

    result_df = result_df.with_columns(
        pl.lit(seq_name).alias("chrom")
    )

    return result_df

def transform_coords(
    df: pl.DataFrame,
    coordinates_valid2raw: dict[str, np.ndarray]
) -> pl.DataFrame:
    """
    transform the coordinates from valid to raw using the coordinate mapping
    Inputs:
        df: pl.DataFrame, the input dataframe with columns ['seqname', 'start', 'end', ...]
        coordinates_valid2raw: dict[str, np.ndarray], the mapping from
    Output:
        pl.DataFrame, the output dataframe with transformed coordinates
    """
    result_start = np.empty(len(df), dtype=np.int32)
    result_end   = np.empty(len(df), dtype=np.int32)

    seqnames = df["chrom"].to_numpy()
    starts   = df["start"].to_numpy()
    ends     = df["end"].to_numpy()

    # group by seqname
    unique_seqs = np.unique(seqnames)

    for seq in unique_seqs:
        mask = (seqnames == seq)

        arr = coordinates_valid2raw[seq]

        result_start[mask] = arr[starts[mask]]
        result_end[mask]   = arr[ends[mask]]

    # write back to dataframe
    df = df.with_columns([
        pl.Series("start", result_start),
        pl.Series("end", result_end),
    ])

    return df

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

def calculate_rotation_invariant_distance(a: str, b: str) -> int:
    """
    distance function for BK-tree
    """
    encoded_a: np.ndarray = encode_seq_to_array(a)
    encoded_b: np.ndarray = encode_seq_to_array(b)
    distance: int = calculate_edit_distance_between_motifs(encoded_a, encoded_b)

    return distance

def calculate_phase_difference(m1: str, m2: str) -> int:
    """
    calculate phase difference between motif1 and motif2, m1 = m2[phase_diff:] + m2[:phase_diff]
    Inputs:
        m1: str
        m2: str
    Outputs:
        phase_diff: int
    """
    is_swap: bool = False
    if len(m1) < len(m2):
        is_swap = True
        m1, m2 = m2, m1

    encoded_m1: np.ndarray = encode_seq_to_array(m1)
    encoded_m2: np.ndarray = encode_seq_to_array(m2)

    score_array, band_argmax_j, _, _, _ = banded_dp_align(
        seq=encoded_m1,
        motif=encoded_m2,
        band_width=len(m2),
        align_to_end=True,
        anchor_row=-1,
        compare_row=-1,
    )

    state = np.argmax(score_array[len(m1) - 1, :])
    best_j = band_argmax_j[len(m1) - 1, state]
    if is_swap:
        phase_diff = len(m2) - (best_j + 1)
    else:
        phase_diff = best_j + 1

    return phase_diff


def _select_ksize_by_scan_coverage(cfg: dict[str, Any]) -> int:
    """
    run a quick vampire scan internally and select the ksize with the highest
    total coverage in the polished_rgns output.
    input:
        cfg: dict, anno configuration (must contain input, threads, job_dir, etc.)
    output:
        best_k: int, the ksize with highest polished_rgns coverage
    """
    from vampire._scan import run_scan

    temp_job_dir = f"{cfg['job_dir']}/auto_scan"
    Path(temp_job_dir).mkdir(parents=True, exist_ok=True)

    scan_cfg = {
        "input": cfg["input"],
        "prefix": f"{temp_job_dir}/auto_scan",
        "job_dir": temp_job_dir,
        "threads": cfg["threads"],
        "debug": False,
        "seq_win_size": 5000000,
        "seq_ovlp_size": 100000,
        "ksize": [17, 13, 9, 5, 3],
        "rolling_win_size": 5,
        "min_smoothness": 50,
        "match_score": cfg["match_score"],
        "mismatch_penalty": cfg["mismatch_penalty"],
        "gap_open_penalty": cfg["gap_open_penalty"],
        "gap_extend_penalty": cfg["gap_extend_penalty"],
        "max_period": 1000,
        "min_score": 50,
        "min_copy": 1.5,
        "secondary": 1.0,
        "format": "brief",
        "skip_cigar": True,
        "skip_report": True,
    }

    run_scan(scan_cfg)

    polished_rgns_dir = Path(temp_job_dir) / "polished_rgns"
    coverage_by_k: dict[int, int] = {}

    for filepath in polished_rgns_dir.glob("window_*.tsv"):
        with open(filepath, "r", newline="", encoding="utf-8") as fi:
            reader = csv.reader(fi, delimiter="\t")
            for row in reader:
                if len(row) < 5:
                    continue
                try:
                    ksizes = ast.literal_eval(row[4])
                    start = int(row[2])
                    end = int(row[3])
                except (ValueError, SyntaxError):
                    continue
                length = end - start + 1
                for k in ksizes:
                    coverage_by_k[k] = coverage_by_k.get(k, 0) + length

    if not cfg.get("debug", False):
        shutil.rmtree(temp_job_dir, ignore_errors=True)

    if not coverage_by_k:
        raise ValueError("no tandem repeats detected in the input sequence, cannot auto-select ksize")

    best_k = max(coverage_by_k, key=coverage_by_k.get)
    logger.info(f"auto-selected k={best_k} (coverage {coverage_by_k[best_k]:,} bp)")
    return best_k


"""
#
# main function for annotating single TR locus among samples
#
"""
def run_anno(cfg: dict[str, Any]) -> None:
    """
    Run the anno function

    Parameters
    ----------
        cfg : dict[str, Any], configuration dictionary

    Returns
    -------
        None

    Generates the following files:
        - <prefix>.log # log file
        - <prefix>.concise.tsv # concise annotation file
        - <prefix>.annotation.tsv # detailed annotation file
        - <prefix>.motif.tsv # motif information file
        - <prefix>.distance.tsv # motif edit distance information file
        - <prefix>.web_summary.html # web summary file
        - <JOB_DIR>/annotation/ # motif annotation records (temp)
    """
    # config
    global JOB_DIR
    JOB_DIR = cfg["job_dir"]
    Path(f"{JOB_DIR}/annotation").mkdir(parents=True, exist_ok=True)
    Path(f"{JOB_DIR}/temp").mkdir(parents=True, exist_ok=True)
    # Set global variables for alignment functions
    global MATCH_SCORE, MISMATCH_PENALTY, GAP_OPEN_PENALTY, GAP_EXTEND_PENALTY
    MATCH_SCORE = cfg["match_score"]
    MISMATCH_PENALTY = cfg["mismatch_penalty"]
    GAP_OPEN_PENALTY = cfg["gap_open_penalty"]
    GAP_EXTEND_PENALTY = cfg["gap_extend_penalty"]
    global SEQ_WIN_SIZE, SEQ_OVLP_SIZE
    SEQ_WIN_SIZE = cfg["seq_win_size"]
    SEQ_OVLP_SIZE = cfg["seq_ovlp_size"]
    THREADS = cfg["threads"]

    global CFG
    CFG = cfg

    # set memory limit
    max_limit: int = min(cfg['resource'] * (1024 ** 3), sys.maxsize)
    resource.setrlimit(resource.RLIMIT_AS, (max_limit, resource.RLIM_INFINITY))

    START_TIME: float = time.time()

    # auto-select ksize using scan polished_rgns coverage
    if cfg.get("auto"):
        logger.info("auto-selecting ksize using scan coverage (activated by --auto)")
        cfg['ksize'] = _select_ksize_by_scan_coverage(cfg)

    # read data
    if not Path(cfg['input']).exists():
        raise FileNotFoundError(cfg['input'])
    with open(cfg['input'], 'r') as fi:
        seq_records: list[SeqIO.SeqRecord] = list(SeqIO.parse(fi, "fasta"))
    logger.info(f"Finished reading fasta file: {cfg['input']}")
    
    # read reference motif set
    if cfg['motif'] == 'base':
        db_path = files("vampire.resources").joinpath("refMotif.fa")
    else:
        db_path = cfg['motif']
    with open(db_path, 'r') as fi:
        motif_records: list[SeqIO.SeqRecord] = list(SeqIO.parse(fi, "fasta"))

    rows: list[dict[str, str]] = []
    for record in motif_records:
        motif_name = record.name
        motif = str(record.seq.upper()) # convert to upper case
        rows.append({
            "motif": motif,
            "ref_seq": motif,
            "name": motif_name,
            "copy_number": 0,
            "source": "database"
        })
    motif_catalog: pl.DataFrame = pl.DataFrame(rows) # columns: motif, ref_seq, name, source
    logger.info("Finished loading reference motif set")
    
    annotation_list = []
    motif2ref_motif = dict()

    # preprocess sequences (remove invalid characters and split into windows)
    seqname2len: dict[str, int] = {record.id: len(record.seq) for record in seq_records}
    seqname2seq: dict[str, str] = {record.id: str(record.seq.upper()) for record in seq_records}
    coordinates_raw2valid: dict[str, np.ndarray] = {}
    coordinates_valid2raw: dict[str, np.ndarray] = {}
    have_invalid: bool = False
    with Pool(processes=THREADS) as pool:
        decompose_tasks: list[tuple[str, int, int, np.ndarray]] = []
        for result in tqdm(
            pool.imap(preprocess_sequence, seq_records, chunksize=1),
            total=len(seq_records),
            desc="Preprocessing"
        ):
            seq_name: str = result["seq_name"]
            coordinates_raw2valid[seq_name] = result["coordinates_raw2valid"]
            coordinates_valid2raw[seq_name] = result["coordinates_valid2raw"]
            have_invalid |= result["have_invalid"]
            decompose_tasks.extend(result["tasks"])

    # de novo get motifs
    if not cfg.get("no_denovo", False):
        motif_catalog_list: list[pl.DataFrame] = [motif_catalog]
        decompose_results: list['Decompose'] = []
        with Pool(processes=THREADS) as pool:
            for result in tqdm(
                pool.imap(decompose_sequence, decompose_tasks, chunksize=1),
                total=len(decompose_tasks),
                desc="Decomposing"
            ):
                decompose_results.append(result)
                motif_catalog_list.append(result.motif_df)
        motif_catalog: pl.DataFrames = pl.concat(motif_catalog_list)
        del motif_catalog_list

    # get canonical motif form
    motif_catalog = motif_catalog.with_columns(
        pl.col("motif").map_elements(canonicalize_motif_str, return_dtype=pl.Utf8)
        .alias("canonical_motif")
    )
    motif_catalog_database: pl.DataFrame = motif_catalog.filter(pl.col("source") == "database")
    motif_catalog_denovo: pl.DataFrame = motif_catalog.filter(pl.col("source") == "denovo")

    # canonical motif deduplication
    motif_catalog_denovo = motif_catalog_denovo.group_by("canonical_motif").agg(
        pl.col("motif").first().alias("motif"),
        pl.col("ref_seq").first().alias("ref_seq"),
        pl.col("name").first().alias("name"),
        pl.col("copy_number").sum().alias("copy_number"),
        pl.col("source").first().alias("source")
    ).select(["motif", "ref_seq", "name", "copy_number", "source", "canonical_motif"])

    # pick top motif
    motif_catalog_denovo = motif_catalog_denovo.sort("copy_number", descending=True).head(cfg["motifnum"])
    motif_catalog = pl.concat([motif_catalog_database, motif_catalog_denovo])

    # motif iteration polishing
    motif_catalog = polish_motif(motif_catalog, cfg)

    motif_catalog = motif_catalog.group_by("motif").agg(
        pl.col("ref_seq").first().alias("ref_seq"),
        pl.col("name").first().alias("name"),
        pl.col("copy_number").max().alias("copy_number"),
        pl.col("source").first().alias("source"),
        pl.col("canonical_motif").first().alias("canonical_motif")
    ).select(["motif", "ref_seq", "name", "copy_number", "source", "canonical_motif"])

    # remove motifs from database if no_denovo or force are set
    if not(cfg.get("no_denovo", False) or cfg.get("force", False)):
        motif_catalog = motif_catalog.filter(pl.col("source") == "denovo")

    if cfg.get("debug", False):
        motif_catalog.write_csv(f"{JOB_DIR}/motif_catalog.tsv", separator="\t", null_value=".")

    # update candidate motif set for annotation
    candidate_motifs: list[str] = motif_catalog["motif"].to_list()
    if len(candidate_motifs) == 0:
        raise RuntimeError("No motif detected. Exit.")
    for seg in decompose_results:
        seg.motif_list = candidate_motifs

    # annoate sequence
    dp_tasks: list[tuple(str, str, pl.DataFrame)] = []
    tmp_df_list: list[pl.DataFrame] = []
    cur_seq_name: str = None

    for seg in tqdm(decompose_results, desc="Annotating"):
        seg: Decompose = annotate_sequence(seg)
        seq_name: str = "_".join(seg.name.split("_")[:-1])
        if seq_name != cur_seq_name:
            if len(tmp_df_list) != 0:
                merged_df = pl.concat(tmp_df_list)
                dp_tasks.append((cur_seq_name, seqname2seq[cur_seq_name], merged_df))
                tmp_df_list = []
                if cfg.get("debug", False):
                    merged_df.write_csv(f"{JOB_DIR}/annotation/{cur_seq_name}.tsv", separator="\t", null_value=".")
            cur_seq_name = seq_name
        tmp_df_list.append(seg.anno_df)
    if len(tmp_df_list) != 0:
        merged_df = pl.concat(tmp_df_list)
        dp_tasks.append((seq_name, seqname2seq[seq_name], merged_df))
        if cfg.get("debug", False):
            merged_df.write_csv(f"{JOB_DIR}/annotation/{cur_seq_name}.tsv", separator="\t", null_value=".")
    del tmp_df_list

    # dynamic programming
    dp_results: list[pl.DataFrame] = []
    with Pool(processes=THREADS) as pool:
        for result in tqdm(
            pool.imap(run_dp, dp_tasks),
            total=len(dp_tasks),
            desc="DPing"
        ):
            dp_results.append(result)

    anno_df: pl.DataFrame = pl.concat(dp_results)
    anno_df = anno_df.with_columns(
        pl.col("chrom").replace_strict(seqname2len).alias("length")
    )    

    # transform coordinates
    if have_invalid:
        anno_df = transform_coords(anno_df, coordinates_valid2raw)

    anno_df = anno_df.with_columns(
        pl.col("start") + 1, # transform to 1-based coordinates
        pl.col("end") + 1
    ).select(["chrom", "length", "start", "end", "motif", "orientation", "sequence", "score", "cigar"])

    # add copy number information
    anno_df = anno_df.with_columns(
        pl.col("motif").map_elements(lambda x: len(x), return_dtype=pl.Int32).alias("motif_length")
    )
    anno_df = anno_df.with_columns(
        pl.struct(["cigar", "motif_length"])
        .map_elements(lambda x: get_copy_number(x["cigar"], x["motif_length"]), return_dtype=pl.Float64)
        .alias("copyNumber")
    )

    # make *.motif.tsv
    motif_df = (
        anno_df
        .group_by("motif")
        .agg(pl.col("copyNumber").sum().round(1).alias("copyNumber"))
    )
    motif_df = motif_df.filter(pl.col("motif").is_not_null()).sort(["copyNumber"], descending=True).with_row_index("id")
    tmp: pl.DataFrame = (
        motif_catalog
        .select(["motif", "name"])
        .with_columns(pl.col("name").alias("label"))
    )
    motif_df = (
        motif_df
        .join(tmp, on="motif", how="left")
        .select(["id", "motif", "copyNumber", "label"])
    )
    motif_df.write_csv(f"{cfg['prefix']}.motif.tsv", separator="\t", null_value=".")
    logger.info(f"Wrote {cfg['prefix']}.motif.tsv")

    # make *.dist.tsv
    id2motif: dict[int, np.ndarray] = {row["id"]: encode_seq_to_array(row["motif"]) for row in motif_df.iter_rows(named=True)}
    id2motif_rc: dict[int, np.ndarray] = {row["id"]: encode_seq_to_array(rc(row["motif"])) for row in motif_df.iter_rows(named=True)}
    motif_num: int = len(id2motif)
    rows: list[dict] = [{
        "target": i,
        "query": j,
        "distance": calculate_edit_distance_between_motifs(id2motif[i], id2motif[j]),
        "sum_copyNumber": motif_df["copyNumber"][i] + motif_df["copyNumber"][j],
        "is_rc": False
    } for i in range(motif_num) for j in range(i + 1, motif_num)]
    rows_rc: list[dict] = [{
        "target": i,
        "query": j,
        "distance": calculate_edit_distance_between_motifs(id2motif[i], id2motif_rc[j]),
        "sum_copyNumber": motif_df["copyNumber"][i] + motif_df["copyNumber"][j],
        "is_rc": True
    } for i in range(motif_num) for j in range(i, motif_num)]
    dist_df: pl.DataFrame = pl.DataFrame(rows + rows_rc)
    dist_df = dist_df.sort(["distance", "sum_copyNumber", "target", "query"]).select(["target", "query", "distance", "is_rc"])
    dist_df.write_csv(f"{cfg['prefix']}.distance.tsv", separator="\t", null_value=".")
    logger.info(f"Wrote {cfg['prefix']}.distance.tsv")

    # make *.annotation.tsv
    anno_df = (
        anno_df
        .join(motif_df.select(["motif", "id"]), on="motif", how="left")
        .with_columns(pl.col("id").alias("motif"))
    )
    anno_df.select(["chrom", "length", "start", "end", "motif", "orientation", "sequence", "score", "cigar"]).write_csv(f"{cfg['prefix']}.annotation.tsv", separator="\t", null_value=".")
    logger.info(f"Wrote {cfg['prefix']}.annotation.tsv")

    # make *.concise.tsv
    concise_df = (
        anno_df
        .group_by("chrom")
        .agg(
            pl.col("length").first().alias("length"),
            pl.col("start").min().alias("start"),
            pl.col("end").max().alias("end"),
            pl.col("motif")
                .drop_nulls()
                .str.join(",")
                .alias("motif"),
            pl.col("orientation")
                .drop_nulls()
                .str.join(",")
                .alias("orientation"),
            pl.col("score").sum().alias("score"),
            pl.col("cigar").str.join("").alias("cigar"),
            pl.col("motif").drop_nulls().last().alias("last_motif"),
            pl.col("copyNumber").sum().round(1).alias("copyNumber"),
        )
    )
    concise_df = concise_df.filter(pl.col("score") >= cfg["min_score"])
    concise_df = concise_df.select(["chrom", "length", "start", "end", "motif", "orientation", "copyNumber", "score", "cigar"])
    concise_df.write_csv(f"{cfg['prefix']}.concise.tsv", separator="\t", null_value=".")
    logger.info(f"Wrote {cfg['prefix']}.concise.tsv")

    END_TIME: float = time.time()
    TIME_USED: float = round(END_TIME - START_TIME, 2)
    logger.info(f"Time used: {TIME_USED:.2f} seconds")

    is_empty: bool = len(anno_df.filter(pl.col("motif").is_not_null())) == 0
    # generate h5ad file
    if not cfg.get("skip_h5ad", False):
        if is_empty:
            logger.warning("No motif is annotated, skip generating h5ad file")
        else:
            import vampire as vp
            adata = vp.anno.pp.read_anno(f"{cfg['prefix']}.annotation.tsv")
            adata.write(f"{cfg['prefix']}.h5ad")
            logger.info(f"Wrote {cfg['prefix']}.h5ad")

    # make web_summary.html
    if not cfg.get("skip_report", False) and not is_empty:            
        # import utils
        from vampire._report_utils import make_stats, make_report
        # add data
        cfg["time_used"] = TIME_USED
        cfg["subcommand"] = "anno"
        # make stats and report
        data: dict[str, Any] = make_stats(cfg)
        make_report(JOB_DIR, str(files("vampire.resources").joinpath("anno_web_summary_template.html")), data)
        # copy web summary file
        web_summary_src = f"{JOB_DIR}/web_summary.html"
        web_summary_dst = f"{cfg["prefix"]}.web_summary.html"
        shutil.copy2(web_summary_src, web_summary_dst)
        logger.info(f"Generated web summary: {web_summary_dst}")

    logger.info("Bye.")

    # copy log file
    shutil.copy2(f"{JOB_DIR}/log.log", f"{cfg['prefix']}.log")
