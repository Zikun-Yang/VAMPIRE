import ast
import csv
import sys
import resource
import shutil
from pathlib import Path
from typing import Any, Iterator
from importlib.resources import files
import networkx as nx
import time
import logging
from tqdm import tqdm
import numba
import ahocorasick

import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl
from pybktree import BKTree
import numpy as np
from Bio import SeqIO   # I/O processing
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

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

logger = logging.getLogger(__name__)

class Decompose:
    def __init__(self, name, sequence, ksize, args_decomp, args_anno, global_kmer_set=None, padding_motif=None):
        self.name = name
        self.sequence = sequence
        self.ksize = ksize
        self.kratio = args_decomp['kratio']
        self.kmin = args_decomp['kmin']
        self.min_similarity = args_anno['annotation_min_similarity']
        self.global_kmer_set = global_kmer_set
        self.padding_motif = padding_motif

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
        kmer_count_int: dict = {}

        # Pad with scan consensus motif so boundary k-mers appear in multiple
        # copies and can form SCCs in the De Bruijn graph.
        # Use adaptive padding: complete partial boundary copies before adding
        # full copies, avoiding artificial junction k-mers when the first or
        # last copy is incomplete.
        seq_for_counting = encoded_seq
        if self.padding_motif:
            encoded_padding = encode_seq_to_array(self.padding_motif)
            if len(encoded_padding) > 0:
                pad_left, pad_right = _make_adaptive_padding(
                    encoded_seq, encoded_padding, self.ksize
                )
                seq_for_counting = np.concatenate([pad_left, encoded_seq, pad_right])

        seq_for_counting_len = len(seq_for_counting)
        if seq_for_counting_len >= k_len:
            for i in range(seq_for_counting_len - k_len + 1):
                w = seq_for_counting[i : i + k_len]
                key = encode_array_to_int(w)
                if key is None:
                    continue
                kmer_count_int[key] = kmer_count_int.get(key, 0) + 1
        
        max_count: int = max(kmer_count_int.values(), default=0)
        min_count: int = max(self.kmin, max_count * self.kratio)
        self.kmer = {
            k: v
            for k, v in kmer_count_int.items()
            if v >= min_count or (self.global_kmer_set and k in self.global_kmer_set)
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

    def find_motif(
        self: 'Decompose',
        period_range: tuple[int, int] | None = None,
        max_revisits: int = 2,
    ) -> pl.DataFrame:
        """
        Length-constrained closed walk enumeration.

        Enumerates closed walks in the De Bruijn graph whose lengths fall within
        [L_min, L_max]. A revisit budget limits how many times each node may appear
        in a single walk, which keeps the search tractable for long motifs while
        still capturing simple sub-repeat structures.

        Parameters
        ----------
        period_range : tuple[int, int] | None
            ``(L_min, L_max)`` allowed closed-walk length in nodes (base pairs).
            If ``None``, defaults to ``(1, 1000)``.
        max_revisits : int
            Maximum number of times a node may be visited in one walk.
            ``1`` means simple cycles only; ``2`` captures one revisit per node.
        """
        if self.motif_df is not None:
            return self.motif_df

        # default range of motif length
        if period_range is None:
            L_min = 1
            L_max = 1000
        else:
            L_min, L_max = period_range

        dbg = self.dbg
        motifs: list[list] = []

        # SCC decomposition
        sccs = list(nx.strongly_connected_components(dbg))

        for scc in sccs:
            if len(scc) < L_min:
                continue

            sub_dbg = dbg.subgraph(scc)
            reverse_sub = sub_dbg.reverse()
            scc_nodes = list(scc)
            node_to_idx = {node: i for i, node in enumerate(scc_nodes)}

            for start in scc_nodes:
                dist_to_start = nx.shortest_path_length(reverse_sub, source=start)

                start_idx = node_to_idx[start]
                visit_counts = [0] * len(scc_nodes)
                visit_counts[start_idx] = 1
                stack = [('enter', start, [start], 0)]

                while stack:
                    action, cur, path, depth = stack.pop()
                    cur_idx = node_to_idx[cur]

                    if action == 'exit':
                        visit_counts[cur_idx] -= 1
                        continue

                    stack.append(('exit', cur, path, depth))

                    if depth > L_max:
                        continue
                    if cur in dist_to_start and depth + dist_to_start[cur] > L_max:
                        continue

                    if start in sub_dbg.successors(cur):
                        next_depth = depth + 1
                        if L_min <= next_depth <= L_max:
                            encoded_node_seq = [decode_int_to_array(node, self.ksize) for node in path]
                            encoded_motif = np.concatenate([seq[(self.ksize - 1):] for seq in encoded_node_seq])
                            motif = decode_array_to_seq(encoded_motif)

                            edge_counts: dict[tuple[int, int], int] = {}
                            for i in range(len(path) - 1):
                                e = (path[i], path[i + 1])
                                edge_counts[e] = edge_counts.get(e, 0) + 1
                            edge_counts[(cur, start)] = edge_counts.get((cur, start), 0) + 1

                            min_weight = min(
                                sub_dbg[u][v]['weight'] / edge_counts[(u, v)]
                                for u, v in edge_counts
                            ) if edge_counts else 0

                            motifs.append([motif, None, 'UNKNOWN', min_weight, "denovo"])

                    if cur == start and depth > 0:
                        continue

                    for succ in sub_dbg.successors(cur):
                        if succ == start:
                            continue
                        succ_idx = node_to_idx[succ]
                        if visit_counts[succ_idx] >= max_revisits:
                            continue
                        visit_counts[succ_idx] += 1
                        stack.append(('enter', succ, path + [succ], depth + 1))

        # deduplication by canonicalization
        # keep one with largest copy number for same canonical motifs
        unique_motifs: dict[str, list] = {}
        for row in motifs:
            motif = row[0]
            cn = row[3]
            canonical = canonicalize_motif_str(motif)
            if canonical not in unique_motifs or cn > unique_motifs[canonical][3]:
                unique_motifs[canonical] = row

        deduped_motifs = list(unique_motifs.values())

        motif_df = pl.DataFrame(
            deduped_motifs,
            schema=["motif", "ref_seq", "name", "copy_number", "source"],
            orient="row"
        )
        motif_df = motif_df.sort("copy_number", descending=True)

        self.motif_df = motif_df
        self.motif_list = motif_df['motif'].to_list()
        return self.motif_df

    def annotate_with_motif(self: 'Decompose', reverse: bool = False) -> pl.DataFrame:
        """
        annotate the sequence with motifs
        Input:
            self: Decompose, the Decompose object (use self.sequence, self.motif_list, self.min_similarity)
            reverse: bool, default is False
        Output:
            pl.DataFrame, the dataframe of annotations
        """
        if self.anno_df is not None:
            return self.anno_df

        # get motifs and max distances
        motifs: list[str] = self.motif_list
        encoded_motifs: list[np.ndarray] = [encode_seq_to_array(m) for m in motifs]
        if reverse:
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
        if reverse and len(motifs_rc) > 0:
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
        result = pl.concat([motif_match_df, motif_rc_match_df]) # ["start", "end", "distance", "score", "cigar", "motif", "orientation"], coordinates are 0-based, end is inclusive

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


def _make_adaptive_padding(
    encoded_seq: np.ndarray,
    encoded_padding: np.ndarray,
    ksize: int,
    n_full: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build left/right padding for k-mer counting.

    The consensus motif phase at each end is inferred with banded alignment
    (calculate_phase_difference): the prefix of the sequence is aligned to the
    consensus and the phase difference gives the missing bases needed to complete
    the partial boundary copy.  Full consensus copies are then appended on the
    outside.

    For the right end, the actual partial copy length is derived from left_phase
    and the sequence length (modulo arithmetic), so that phase inference uses
    only the partial copy — not the last m bases which include the previous
    full copy.

    Returns
    -------
    pad_left, pad_right
    """
    m = len(encoded_padding)
    if m == 0:
        return np.empty(0, dtype=encoded_seq.dtype), np.empty(0, dtype=encoded_seq.dtype)

    full_copies: np.ndarray = np.tile(encoded_padding, n_full)

    if len(encoded_seq) < m:
        return full_copies, full_copies

    padding_str: str = decode_array_to_seq(encoded_padding)
    n = len(encoded_seq)

    # Left side: complete the first partial copy, then add full copies.
    prefix_str: str = decode_array_to_seq(encoded_seq[:m])
    left_phase: int = calculate_phase_difference(prefix_str, padding_str)
    left_missing = encode_seq_to_array(padding_str[:left_phase])
    pad_left = np.concatenate([full_copies, left_missing])

    # Right side: determine the actual partial copy length at the end.
    len_last_copy: int = (left_phase + n) % m
    if len_last_copy == 0:
        # Sequence ends exactly at a motif boundary — no missing bases.
        right_missing = np.empty(0, dtype=encoded_seq.dtype)
    else:
        # Take only the last partial copy for phase inference.
        partial_suffix = encoded_seq[-len_last_copy:]
        partial_suffix_str = decode_array_to_seq(partial_suffix)

        # Find which rotation of the consensus best matches this partial copy.
        best_phase = 0
        best_dist = len_last_copy + 1
        for phase in range(m):
            rotated = np.concatenate([encoded_padding[phase:], encoded_padding[:phase]])
            expected = rotated[:len_last_copy]
            dist = calculate_edit_distance_between_motifs(partial_suffix, expected)
            if dist < best_dist:
                best_dist = dist
                best_phase = phase

        right_missing_start = (best_phase + len_last_copy) % m
        if right_missing_start == 0:
            right_missing = np.empty(0, dtype=encoded_seq.dtype)
        else:
            right_missing = encoded_padding[right_missing_start:]

    pad_right = np.concatenate([right_missing, full_copies])

    return pad_left, pad_right


def rc(seq: str) -> str:
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N':'N'}
    return ''.join(complement[base] for base in reversed(seq))


def _count_exact_motif_occurrences(
    sequences: list[str],
    motifs: list[str],
    include_rc: bool = False,
) -> dict[str, int]:
    """
    Count exact substring occurrences of each motif across all sequences.

    Uses an Aho-Corasick automaton for linear-time multi-pattern matching.
    Overlapping matches are counted (e.g. ``"AAA"`` in ``"AAAAAA"`` yields 4).

    Because the motif catalog stores one arbitrary rotation of each canonical
    motif, all cyclic rotations (and reverse-complement rotations when
    ``include_rc`` is True) are indexed. The maximum count across rotations of
    the same original motif is returned, which recovers the true copy number
    regardless of the phase present in the sequence.

    Parameters
    ----------
    sequences : list[str]
        Input sequences.
    motifs : list[str]
        Motifs to query.
    include_rc : bool, optional
        If True, also count occurrences of the reverse complement of each motif.

    Returns
    -------
    dict[str, int]
        Mapping from each input motif to its total exact occurrence count.
    """
    A = ahocorasick.Automaton()
    rotation_to_idx: dict[str, int] = {}
    indexed_rotations: list[str] = []
    rotation_to_originals: dict[str, list[str]] = {}

    def _register(rotation: str, original: str) -> None:
        """Index a rotation and record which original motifs it represents."""
        if rotation not in rotation_to_idx:
            rotation_to_idx[rotation] = len(indexed_rotations)
            indexed_rotations.append(rotation)
            rotation_to_originals[rotation] = [original]
            A.add_word(rotation, rotation_to_idx[rotation])
        elif original not in rotation_to_originals[rotation]:
            rotation_to_originals[rotation].append(original)

    for motif in motifs:
        if not motif:
            continue
        seen_rotations: set[str] = set()
        for shift in range(len(motif)):
            rotation = motif[shift:] + motif[:shift]
            if rotation in seen_rotations:
                continue
            seen_rotations.add(rotation)
            _register(rotation, motif)
        if include_rc:
            rc_motif = rc(motif)
            for shift in range(len(rc_motif)):
                rotation = rc_motif[shift:] + rc_motif[:shift]
                if rotation in seen_rotations:
                    continue
                seen_rotations.add(rotation)
                _register(rotation, motif)

    if not indexed_rotations:
        return {motif: 0 for motif in motifs}

    A.make_automaton()

    rotation_counts = [0] * len(indexed_rotations)
    for seq in sequences:
        for _end_index, idx in A.iter(seq):
            rotation_counts[idx] += 1

    # In a tandem repeat all copies share the same phase, so only one rotation
    # matches at each position. Taking the max across rotations recovers the
    # copy number regardless of which rotation was stored in the catalog.
    best_counts: dict[str, int] = {motif: 0 for motif in motifs if motif}
    for idx, rotation in enumerate(indexed_rotations):
        for original in rotation_to_originals[rotation]:
            if rotation_counts[idx] > best_counts[original]:
                best_counts[original] = rotation_counts[idx]

    result: dict[str, int] = {}
    for motif in motifs:
        result[motif] = best_counts.get(motif, 0)

    return result


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

    seq: str = str(record.seq.upper())
    encoded_seq: np.ndarray = encode_seq_to_array(seq)
    seq_len_with_N: int = len(seq)
    SEQ_STEP_SIZE: int = SEQ_WIN_SIZE - SEQ_OVLP_SIZE

    invalid_mask: np.ndarray = encoded_seq == -1

    valid2raw: np.ndarray = np.where(~invalid_mask)[0].astype(np.int32)

    raw2valid: np.ndarray = np.full(seq_len_with_N, -1, dtype=np.int32)
    raw2valid[valid2raw] = np.arange(len(valid2raw), dtype=np.int32)

    if invalid_mask.any():
        invalid_pos: list[int] = np.where(invalid_mask)[0].tolist()
        invalid_char: list[str] = sorted(set(seq[i] for i in invalid_pos))
        logger.warning(
            f"Invalid characters found in sequence {seq_name}: {invalid_char}"
        )

    # filter invalid chars
    encoded_seq = encoded_seq[encoded_seq != -1]
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

    # Global k-mer census: count kmers on the full valid sequence (no filtering)
    kmer_counts: dict[int, int] = {}
    k_len = CFG["ksize"] + 1
    if encoded_seq_len >= k_len:
        for i in range(encoded_seq_len - k_len + 1):
            w = encoded_seq[i : i + k_len]
            key = encode_array_to_int(w)
            if key is not None:
                kmer_counts[key] = kmer_counts.get(key, 0) + 1

    return {
        "seq_name": seq_name,
        "length": encoded_seq_len,
        "sequence": decode_array_to_seq(encoded_seq),
        "have_invalid": bool(invalid_mask.any()),
        "coordinates_raw2valid": raw2valid,
        "coordinates_valid2raw": valid2raw,
        "tasks": tasks,
        "kmer_counts": kmer_counts,
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
        'Decompose'
    """
    seq_name, pid, total_task, sequence = task

    decomp_opts = {
        "kratio": CFG["kratio"],
        "kmin": CFG["kmin"]
    }
    anno_opts = {"annotation_min_similarity": CFG["annotation_min_similarity"]}
    seg = Decompose(
        name = f"{seq_name}_{pid}",
        sequence = sequence,
        ksize = CFG["ksize"],
        args_decomp = decomp_opts,
        args_anno = anno_opts,
        global_kmer_set = CFG.get("global_kmer_set"),
        padding_motif = CFG.get("padding_motif"),
    )
    filtered_kmers: dict[int, int] = seg.count_kmers()

    if not filtered_kmers:
        logger.debug(
            f"Skip decomposition {seq_name}: no valid high-frequency k-mers"
        )
        return seg

    seg.build_graph()
    period_range = CFG.get("period_range")
    seg.find_motif(
        period_range=period_range,
        max_revisits=CFG.get("max_revisits", 2),
    )

    return seg

def do_not_decompose_sequence(
    task: tuple[str, int, int, np.ndarray]
) -> 'Decompose':
    """
    Decompose sequence into motifs
    Input:
        task: tuple[str, int, int, np.ndarray]
    Output:
        'Decompose'
    """
    seq_name, pid, total_task, sequence = task

    decomp_opts = {
        "kratio": CFG["kratio"],
        "kmin": CFG["kmin"]
    }
    anno_opts = {"annotation_min_similarity": CFG["annotation_min_similarity"]}
    seg = Decompose(
        name = f"{seq_name}_{pid}",
        sequence = sequence,
        ksize = CFG["ksize"],
        args_decomp = decomp_opts,
        args_anno = anno_opts,
        global_kmer_set = CFG.get("global_kmer_set"),
        padding_motif = CFG.get("padding_motif"),
    )

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
    n_not_matched: int = 0 # use as the suffix of motif name
    
    # sort by copy_number (descending) to process high-confidence motifs first
    motif_catalog_denovo = motif_catalog_denovo.sort("copy_number", descending=True)
    
    for idx in range(len(motif_catalog_denovo)):
        row = motif_catalog_denovo.row(idx, named = True)
        motif: str = row["motif"]
        motif_len: int = len(motif)
        canonical_form: str = row["canonical_motif"]
        finding_min_similarity: float = cfg.get("finding_min_similarity", 0.5)
        max_distance: int = int((1 - finding_min_similarity) * len(motif))
        
        min_distance: int = 1_000_000
        best_motif: str = canonical_form
        best_ref: str = canonical_form
        name: str = f"unknown_{n_not_matched + 1}"
        is_matched: bool = False

        logger.debug(f"Polishing motif {idx}: {motif}")
        new_row: dict[str, Any] = {
            "motif": canonical_form,
            "ref_seq": canonical_form,
            "name": name,
            "copy_number": row["copy_number"],
            "source": "denovo",
            "canonical_motif": canonical_form
        }
        new_row_list: list[dict[str, Any]] = [new_row]
        
        # search canonical form
        matches = tree.find(canonical_form, max_distance)
        logger.debug("Trying match canonical form ...")
        for dist, ref_motif in matches:
            if dist < min_distance:
                min_distance = dist
                ref_row = motif_catalog_database.row(ref_to_idx[ref_motif], named = True)
                best_ref = ref_motif
                phase = calculate_phase_difference(best_ref, motif)
                best_motif = motif[phase:] + motif[: phase]
                name = ref_row["name"]
                is_matched = True
                logger.debug(f"╵-> Succesfully matched! ref_motif: {ref_motif}\n")
        
        # also check reverse complement canonical form
        if not is_matched:
            logger.debug("Trying match reverse complementary form ...")
            motif_rc = rc(canonical_form)
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
                    logger.debug(f"╵-> Succesfully matched reverse complementary form! ref_motif: {ref_motif}")
        
        # update match information
        if is_matched:
            new_row["motif"] = best_motif
            new_row["ref_seq"] = best_ref
            new_row["name"] = name
            new_row_list: list[dict[str, Any]] = [new_row]

        # concatemer processing - only process if already matched
        is_cut: bool = False
        if is_matched:
            logger.debug("Trying cut concatemer ...")
            cut_units, is_cut = _try_cut(
                motif = canonical_form, 
                basic_unit = best_ref, 
                min_similarity = 0.8,
                min_remaining_ratio = 0.5,
            )
            if is_cut:
                logger.debug(f"Cut motif {motif} to {cut_units}")
                new_row_list = []
                for unit in cut_units:
                    # add cut motif to tree for subsequent processing
                    canonical_unit = canonicalize_motif_str(unit)
                    if canonical_unit not in ref_to_idx:
                        tree.add(canonical_unit)
                        ref_to_idx[canonical_unit] = len(ref_to_idx)
                    new_row = {
                        "motif": canonical_unit,
                        "ref_seq": best_ref,
                        "name": name,
                        "copy_number": row["copy_number"],
                        "source": "denovo",
                        "canonical_motif": canonical_unit
                    }
                    new_row_list.append(new_row)
                new_row_df = pl.DataFrame(new_row_list)
                motif_catalog_database = motif_catalog_database.vstack(new_row_df)
                logger.debug("╵-> Succesfully cut!")
        
        # not matched
        if not is_matched:
            if canonical_form not in ref_to_idx:
                logger.debug(f"### Motif {idx} is not registered, pushing it into the tree\n")
                tree.add(canonical_form)
                ref_to_idx[canonical_form] = len(ref_to_idx)
                new_row_df = pl.DataFrame(new_row_list)
                motif_catalog_database = motif_catalog_database.vstack(new_row_df)
                n_not_matched += 1

        polished_rows.extend(new_row_list)
    
    # Combine database and polished denovo motifs
    if polished_rows:
        polished_denovo_df = pl.DataFrame(polished_rows)
        motif_catalog_database = motif_catalog_database.filter(pl.col("source") == "database")
        result_catalog = pl.concat([motif_catalog_database, polished_denovo_df])
    else:
        result_catalog = motif_catalog_database
    
    return result_catalog

def _try_cut(
    motif: str,
    basic_unit: str,
    min_similarity: float = 0.8,
    min_remaining_ratio: float = 0.5,
) -> tuple[list[str], bool]:
    """
    Attempt to decompose a motif using a shorter basic repeat unit.

    The motif is aligned against repeated copies of ``basic_unit`` using a
    cyclic banded alignment. The resulting alignment is split into unit-sized
    blocks corresponding to individual copies of the basic unit.

    If at least one block matches the basic unit with sufficient similarity,
    blocks containing sequence differences (substitutions, insertions, or
    deletions) are extracted as candidate submotifs. Very short residual
    fragments near the end of the motif are preserved as a single uncut tail
    rather than being split further.

    Parameters:
    motif : str
        Candidate motif to be decomposed.
    basic_unit : str
        Putative elementary repeat unit.
    min_similarity : float, optional
        Minimum similarity required for the best-aligned unit block. If no
        block reaches this threshold, the motif is left unchanged.
    min_remaining_ratio : float, optional
        Minimum remaining length (relative to ``basic_unit``) required to
        continue splitting. Shorter terminal fragments are kept intact.

    Returns:
    tuple[list[str], bool]
        A tuple containing:

        - List of extracted submotifs.
        - Whether any cutting operation was performed.
    """
    cut_units: list[str] = []
    seq_len: int = len(motif)
    motif_len: int = len(basic_unit)
    encoded_motif: np.ndarray = encode_seq_to_array(motif)
    encoded_unit: np.ndarray = encode_seq_to_array(basic_unit)

    # banded alignment to decompose
    score_array, band_argmax_j, trace_M, trace_I, trace_D = banded_dp_align(
        seq = encoded_motif,
        motif = encoded_unit,
        band_width = len(motif),
        align_to_end = True,
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
        seq = encoded_motif,
        motif = encoded_unit
    )

    # data structure: (start, end, distance, score, cigar)
    unit_blocks: list[tuple[int, int, int, int, str]] = split_ops(ops, len(basic_unit), len(basic_unit))
    min_distance = min(block[2] for block in unit_blocks)

    # if the best block is still too dissimilar, give up cutting
    if min_distance / len(basic_unit) > (1 - min_similarity):
        return [motif], False

    is_cut: bool = False
    for block in unit_blocks:
        start, end, distance, score, cigar = block
        # if too short after ccutting, keep the tail uncut
        if (seq_len - end + 1) / len(basic_unit) < min_remaining_ratio:
            cut_units.append(motif[start: ])
            break
        # cut the motif and canonicalize the remaining part
        if distance != 0:
            cut_units.append(motif[start: end + 1])
        is_cut = True

    return cut_units, is_cut

def annotate_sequence(
    seg: Decompose,
    reverse: bool,
) -> Decompose:
    """
    Annotate sequence with motifs
    Input:
        seg: Decompose
        reverse: bool
    Output:
        seg: Decompose
    """
    seg.annotate_with_motif(reverse)

    anno_df: pl.DataFrame = seg.anno_df
    pid: int = int(seg.name.split("_")[-1])
    anno_df = anno_df.with_columns(
        (pl.col("start") + (pid - 1) * (SEQ_WIN_SIZE - SEQ_OVLP_SIZE)).alias("start"),
        (pl.col("end") + (pid - 1) * (SEQ_WIN_SIZE - SEQ_OVLP_SIZE)).alias("end")
    )
    anno_df = anno_df.sort("end")
    ###logger.debug(f"Finished annotation: {seg.name}")
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
            # solve partial copy
            ops_list: list[str] = ops[ops_start: i + 1]
            leading_del: int = 0
            for op in ops_list:
                if op == "D":
                    leading_del += 1
                else:
                    break
            trailing_del: int = 0
            for op in reversed(ops_list):
                if op == "D":
                    trailing_del += 1
                else:
                    break
            MIN_PARTIAL = max(2, round(0.1 * m)) # minimum missing end length
            if leading_del >= MIN_PARTIAL:
                distance -= leading_del
                cur_score += leading_del * GAP_EXTEND_PENALTY
            if trailing_del >= MIN_PARTIAL:
                distance -= trailing_del
                cur_score += trailing_del * GAP_EXTEND_PENALTY

            if distance <= max_distance:
                split_results.append((start, cur_pos, distance, cur_score, ops_to_cigar(ops[ops_start: i + 1], m))) # 0-based coordinates; end is inclusive
            start = cur_pos + 1
            ops_start = i + 1
            distance = 0
            cur_phase = -1
            cur_score = 0
    
    if cur_phase >= 0:
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
    schema_dict: dict = {
        "start": pl.Int64,
        "end": pl.Int64,
        "distance": pl.Int64,
        "score": pl.Int64,
        "cigar": pl.Utf8,
        "motif": pl.Utf8,
        "orientation": pl.Utf8,
        "sequence": pl.Utf8,
    }
    result_df: pl.DataFrame = pl.DataFrame(rows, schema=schema_dict, orient="row").sort("start")

    # add segments that are not annotated with motifs
    if result_df.shape[0] == 0:
        rows.append(
            {
                "start": 0,
                "end": seq_len - 1,
                "distance": None,
                "score": - GAP_OPEN_PENALTY - (seq_len - 1) * GAP_EXTEND_PENALTY,
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
                        "score": - GAP_OPEN_PENALTY - (end - start - 2) * GAP_EXTEND_PENALTY,
                        "cigar": f"{end - start - 1}N",
                        "motif": None,
                        "orientation": None,
                        "sequence": seq[start + 1: end] # here is not end - 1 because the end is the inclusive start of next annotated region
                    }
                )
    if len(rows) > row_len:
        result_df = pl.DataFrame(                                                                                                    
            rows,                                                                                                                    
            schema=schema_dict,
            orient="row"                                                                                                             
        ).sort("start")

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
        phase_diff: int, 0-based phase difference; 0 means m1 and m2 are on the same phase, 1 means m1 is shifted by 1 base compared to m2, etc.
            if m1 and m2 are the same motif with different phase, return the positive phase difference;
            if m1 and m2 are the same motif with the same phase, return 0; if m1 and m2 are different motifs, return the phase difference that can align them best
    """
    import edlib
    import re

    def _longest_match_from_cigar(cigar: str | None) -> int:
        if cigar is None:
            return 0
        best = 0
        cur = 0
        for length_str, op in re.findall(r"(\d+)([=XIDM])", cigar):
            length = int(length_str)
            if op == "=":
                cur += length
                if cur > best:
                    best = cur
            else:
                cur = 0
        return best

    m = len(m2)
    if m == 0:
        return 0

    best_key: tuple[int, int, int] | None = None
    best_phase = 0

    for phase in range(m):
        rotated = m2[phase:] + m2[:phase]
        alignment = edlib.align(rotated, m1, task="path")
        edit_distance = alignment["editDistance"]
        longest_match = _longest_match_from_cigar(alignment.get("cigar"))

        # Smallest edit distance first, then longest continuous match,
        # then smallest phase for deterministic tie-breaking.
        key = (edit_distance, -longest_match, phase)
        if best_key is None or key < best_key:
            best_key = key
            best_phase = phase

    return best_phase

"""
#
# ksize parameter selecting functions
#
"""
def _compute_periodicity_from_distance(
    distance_dir: Path,
    k: int,
    expected_period: float | None = None,
) -> dict[str, float] | None:
    """
    Load all distance arrays for a given k-mer size and compute periodicity metrics.

    If expected_period is provided, the metrics focus on how tightly the distance
    distribution clusters around that expected motif length (near_ratio, near_cv,
    mean_dev).  Otherwise, fall back to the global dominant-period metrics.
    """
    files = list(distance_dir.glob(f"*-{k}.npy"))
    if not files:
        return None

    total_positions = 0
    all_valid_dists: list[np.ndarray] = []

    for f in files:
        dist = np.load(f)
        total_positions += dist.size
        valid = dist[~np.isnan(dist)]
        if valid.size > 0:
            all_valid_dists.append(valid)

    if not all_valid_dists:
        return {
            "valid_ratio": 0.0,
            "near_ratio": 0.0,
            "near_cv": float("inf"),
            "mean_dev": float("inf"),
            "mean_dist": 0.0,
        }

    arr = np.concatenate(all_valid_dists)
    valid_ratio = arr.size / total_positions if total_positions > 0 else 0.0

    if expected_period and expected_period > 0:
        # Focus on distances near the expected motif period.
        window = expected_period * 0.1
        deviations = np.abs(arr - expected_period)
        near_mask = deviations <= window
        near = arr[near_mask]

        near_ratio = float(near.size / arr.size) if arr.size > 0 else 0.0
        near_std = float(near.std()) if near.size > 1 else float("inf")
        near_cv = near_std / expected_period
        mean_dev = float(deviations.mean()) if arr.size > 0 else float("inf")

        return {
            "valid_ratio": valid_ratio,
            "near_ratio": near_ratio,
            "near_cv": near_cv,
            "mean_dev": mean_dev,
            "mean_dist": float(arr.mean()),
        }

    # Fallback: global dominant-period metrics when no expected period is known.
    int_dists = np.round(arr).astype(int)
    unique, counts = np.unique(int_dists, return_counts=True)
    mode_count = int(counts.max())
    period_strength = mode_count / arr.size

    mean_d = float(arr.mean())
    std_d = float(arr.std())
    cv = std_d / mean_d if mean_d > 0 else float("inf")

    return {
        "valid_ratio": valid_ratio,
        "period_strength": period_strength,
        "cv": cv,
        "mean_dist": mean_d,
    }


def _select_ksize_by_scan_coverage(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    run a quick vampire scan internally and select the ksize with the highest
    total coverage in the polished_rgns output.
    input:
        cfg: dict, anno configuration (must contain input, threads, job_dir, etc.)
    output:
        cfg: dict, with best_k (int) the ksize with highest polished_rgns coverage
    """
    import subprocess
    import sys

    candidate_k = [101, 51, 31, 25, 17, 13, 11, 9, 7, 5, 3]

    prefix = f"{cfg['job_dir']}/auto_scan"
    temp_job_dir = f"{cfg['job_dir']}/auto_scan_temp"
    Path(temp_job_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "vampire.main",
        "scan",
        cfg["input"],
        prefix,
        "--job", temp_job_dir,
        "--threads", str(cfg["threads"]),
        "--seq-win-size", "5000000",
        "--seq-ovlp-size", "100000",
        "--ksize", ",".join(str(k) for k in candidate_k),
        "--match-score", str(cfg["match_score"]),
        "--mismatch-penalty", str(cfg["mismatch_penalty"]),
        "--gap-open-penalty", str(cfg["gap_open_penalty"]),
        "--gap-extend-penalty", str(cfg["gap_extend_penalty"]),
        "--skip-report",
        "--skip-cigar",
        "--debug",
        "--format", "brief",
    ]
    
    env = os.environ.copy()
    env["VAMPIRE_SAVE_DISTANCE"] = "1"

    logger.debug(f"Running auto-scan: {' '.join(cmd)}")
    logger.info(f"Selecting k from {candidate_k}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Auto-scan failed (exit {result.returncode}):\n{result.stderr}"
        )

    # check kmin validity
    scan_result = pl.read_csv(f"{cfg['job_dir']}/auto_scan.tsv", separator="\t")
    cn = scan_result["copyNumber"]
    ratio_below = (cn < cfg["kmin"]).mean()
    if ratio_below > 0.5:
        median_cn = cn.median()
        new_kmin = max(1, int(median_cn))

        logger.warning(
            f"{int((cn < cfg['kmin']).sum())} samples below kmin={cfg['kmin']} "
            f"(ratio={ratio_below:.2%}). "
            f"Adjusting kmin -> {new_kmin} (median={median_cn})."
        )

        cfg["kmin"] = new_kmin

    # Derive a consensus motif from scan to pad sequences during k-mer counting.
    # Padding makes boundary k-mers appear in multiple copies so they form SCCs
    # instead of being filtered out.
    if "motif" in scan_result.columns and "copyNumber" in scan_result.columns and not scan_result.is_empty():
        try:
            scan_with_can = scan_result.with_columns(
                pl.col("copyNumber").cast(pl.Float64, strict=False)
            ).with_columns(
                pl.col("motif")
                .map_elements(canonicalize_motif_str, return_dtype=pl.Utf8)
                .alias("canonical_motif")
            )
            agg = scan_with_can.group_by("canonical_motif").agg(
                pl.col("copyNumber").sum().alias("total_cn")
            )
            best_can = agg.sort("total_cn", descending=True).item(0, "canonical_motif")
            best_repr = (
                scan_with_can.filter(pl.col("canonical_motif") == best_can)
                .sort(pl.col("motif").str.len_chars(), descending=True)
                .item(0, "motif")
            )
            cfg["padding_motif"] = best_repr
        except Exception:
            pass

    # estimate motif length
    motif_len_list = scan_result["motif"].drop_nulls().str.len_chars()
    motif_len_list = motif_len_list.filter(motif_len_list > 0)
    period_range = None
    expected_period = None
    if len(motif_len_list) > 0:
        # Normalize integer-multiple motif lengths (dimers, trimers, etc.) to the
        # fundamental period so they do not inflate the estimated range.
        lengths = motif_len_list
        unique_sorted = sorted({int(x) for x in lengths if x > 0})
        if len(unique_sorted) > 1:
            tolerance = 0.1
            normalized: list[int] = []
            for L in lengths:
                L = int(L)
                fundamental = L
                for d in unique_sorted:
                    if d >= L:
                        break
                    ratio = L / d
                    k = round(ratio)
                    if k > 1 and abs(ratio - k) <= tolerance:
                        fundamental = d
                normalized.append(fundamental)
            lengths = pl.Series(normalized)

        # Use the median normalized motif length as the expected period for
        # selecting the k-size whose distance-derived dominant period matches best.
        expected_period = float(lengths.median())

        # calculate statistics
        q25 = int(lengths.quantile(0.25))
        q75 = int(lengths.quantile(0.75))

        # set range based on IQR
        iqr = q75 - q25
        margin = max(int(1.5 * iqr), 2)

        L_min = max(1, int(q25 - margin))
        L_max = int(q75 + margin)

        L_min = L_min if cfg['lmin'] <= 0 else cfg['lmin']
        L_max = L_max if cfg['lmax'] <= 0 else cfg['lmax']

        period_range: tuple[int, int] = (L_min, L_max)

    if period_range:
        cfg["period_range"] = period_range
        logger.info(f"Estimated motif period range: {period_range[0]}-{period_range[1]} bp")

    # Compute periodicity metrics from saved distance arrays
    distance_dir = Path(temp_job_dir) / "distance"
    periodicity_by_k: dict[int, dict[str, float]] = {}
    if distance_dir.exists():
        for k in candidate_k:
            per = _compute_periodicity_from_distance(distance_dir, k, expected_period)
            if per is not None:
                periodicity_by_k[k] = per

    if cfg.get("debug", False):
        shutil.rmtree(temp_job_dir, ignore_errors=True)

    if not periodicity_by_k:
        raise ValueError("No tandem repeats detected in the input sequence, cannot auto-select ksize")

    for k, per in periodicity_by_k.items():
        logger.debug(f"  k={k}: {per}")

    # Select k with the smallest mean deviation from expected period.
    if not cfg.get("no_auto"):
        mean_devs: dict[int, float] = {}
        for k, per in periodicity_by_k.items():
            mean_devs[k] = per.get("mean_dev", float("inf")) ### * (1 - per.get("valid_ratio", float("inf")))

        best_k = min(mean_devs, key=mean_devs.get)
        best_per = periodicity_by_k[best_k]
        logger.info(
            f"Auto-selected k={best_k} "
            f"(expected_period={expected_period:.1f}, "
            f"mean_dev/valid_ratio={best_per.get('mean_dev', float('inf')):.2f})"
        )
        cfg["ksize"] = best_k
    else:
        logger.info(f"Skipping k-size auto-selection due to --no-auto (using k={cfg.get('ksize')})")

    return cfg

"""
#
# make_raw function: use raw motif to re-annotate the sequences
#
"""
def make_raw(
    anno_df: pl.DataFrame,
    concise_df: pl.DataFrame,
    motif_df: pl.DataFrame,
    dist_df: pl.DataFrame,
    match_score: int,
    mismatch_penalty: int,
    gap_open_penalty: int,
    gap_extend_penalty: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Recalculate annotation results using the observed aligned sequences as motifs.

    Each non-gap annotated block is converted to a canonicalized motif derived
    from its real sequence. Reverse-complement blocks are normalized to the
    forward strand before canonicalization. Because the resolved motif equals
    the observed sequence (up to rotation), the alignment CIGAR is replaced by
    a perfect-match CIGAR and each block contributes exactly one copy.
    """
    def _forward_sequence(x: dict) -> str | None:
        seq = x["sequence"]
        if seq is None:
            return None
        return rc(seq) if x["orientation"] == "-" else seq

    def _align_phase(ref: str, query: str) -> str:
        phase = calculate_phase_difference(ref, query)
        adj_query = query[phase:] + query[: phase]
        return adj_query

    global MATCH_SCORE, MISMATCH_PENALTY, GAP_OPEN_PENALTY, GAP_EXTEND_PENALTY
    MATCH_SCORE = match_score
    MISMATCH_PENALTY = mismatch_penalty
    GAP_OPEN_PENALTY = gap_open_penalty
    GAP_EXTEND_PENALTY = gap_extend_penalty

    id2label: dict[int, str] = {
        int(motif_id): label
        for motif_id, label in zip(motif_df["id"], motif_df["label"])
        if label is not None
    }
    max_id: int = max(id2label.keys())
    id2label[max_id + 1] = "skipped"
    anno_df = anno_df.with_columns(
        pl.when(pl.col("motif").is_null())
        .then(pl.lit(max_id) + 1)
        .otherwise(pl.col("motif").cast(pl.Int32))
        .alias("motif")
    )

    # Preserve original label for propagation to raw motifs
    anno_df = anno_df.with_columns(
        pl.col("motif")
        .map_elements(
            lambda x: id2label.get(x),
            return_dtype=pl.Utf8,
        )
        .alias("label")
    )

    # include sequences marked as gap before
    anno_df = anno_df.with_columns(
        pl.col("orientation").fill_null("+").alias("orientation"),
        pl.col("sequence").str.len_chars().alias("motif_length"),
    )
    # reverse sequence if orientation is "-"
    anno_df = anno_df.with_columns(
        pl.struct(["sequence", "orientation"])
        .map_elements(_forward_sequence, return_dtype=pl.Utf8)
        .alias("forward_sequence")
    )
    # canonicalize the motif
    anno_df = anno_df.with_columns(
        pl.col("forward_sequence")
        .map_elements(canonicalize_motif_str, return_dtype=pl.Utf8)
        .alias("motif")
    )
    # update score and cigar
    anno_df = anno_df.with_columns(
        pl.col("motif_length").map_elements(lambda x: f"{x}=/", return_dtype=pl.Utf8).alias("cigar"),
        (pl.col("motif_length") * MATCH_SCORE).alias("score"),
    )

    anno_df = anno_df.with_columns(
        pl.lit(1.0).alias("copyNumber")
    )

    # get motif dataframe
    motif_df = (
        anno_df.group_by(["motif", "label"])
        .agg(pl.col("copyNumber").sum().round(1).alias("copyNumber"))
        .sort("copyNumber", descending=True)
        .with_row_index("id")
        .select(["id", "motif", "copyNumber", "label"])
    )
    if "id" in anno_df.columns:
        anno_df = anno_df.drop("id")
    anno_df = anno_df.join(
        motif_df.select(["motif", "id"]),
        on="motif",
        how="left"
    ).with_columns(
        pl.col("id").alias("motif")
    ).drop("id")

    # get reference motif
    mode_length: int = int(anno_df["motif_length"].mode()[0])
    ref_id: int = (
        anno_df
        .filter(pl.col("motif_length") == mode_length)
        .group_by("motif")
        .agg(pl.col("copyNumber").sum().alias("copyNumber"))
        .sort("copyNumber", descending=True)
        .limit(1)
        .select("motif")
        .item()
    )
    ref_motif: str = motif_df.filter(pl.col("id") == ref_id).select("motif").item()
    logger.debug(f"Use motif {ref_id} {ref_motif} ({mode_length} bp) to align phase")
    # align phase
    motif_df = motif_df.with_columns(
        pl.col("motif").map_elements(lambda x: _align_phase(ref_motif, x), return_dtype=pl.Utf8).alias("motif")
    )

    # Build distance matrix from the new motif catalog.
    encoded_motifs: dict[int, np.ndarray] = {
        row["id"]: encode_seq_to_array(row["motif"])
        for row in motif_df.iter_rows(named=True)
    }
    encoded_motifs_rc: dict[int, np.ndarray] = {
        row["id"]: encode_seq_to_array(rc(row["motif"]))
        for row in motif_df.iter_rows(named=True)
    }
    motif_num: int = len(encoded_motifs)
    rows: list[dict] = [{
        "target": i,
        "query": j,
        "distance": calculate_edit_distance_between_motifs(encoded_motifs[i], encoded_motifs[j]),
        "sum_copyNumber": motif_df["copyNumber"][i] + motif_df["copyNumber"][j],
        "is_rc": False
    } for i in range(motif_num) for j in range(i + 1, motif_num)]
    rows_rc: list[dict] = [{
        "target": i,
        "query": j,
        "distance": calculate_edit_distance_between_motifs(encoded_motifs[i], encoded_motifs_rc[j]),
        "sum_copyNumber": motif_df["copyNumber"][i] + motif_df["copyNumber"][j],
        "is_rc": True
    } for i in range(motif_num) for j in range(i, motif_num)]
    dist_df: pl.DataFrame = pl.DataFrame(rows + rows_rc)
    dist_df = dist_df.sort(["distance", "sum_copyNumber", "target", "query"]).select(["target", "query", "distance", "is_rc"])

    # Build concise summary.
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
    # select and reorder columns
    anno_df = anno_df.select([
        "chrom", "start", "end", "motif", "orientation", 
        "sequence", "copyNumber", "score", "cigar"
    ])
    concise_df = concise_df.select([
        "chrom", "length", "start", "end", "motif", "orientation",
        "copyNumber", "score", "cigar"
    ])
    motif_df = motif_df.select([
        "id", "motif", "copyNumber", "label"
    ])

    return anno_df, concise_df, motif_df, dist_df

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
    CFG["max_revisits"] = 5

    # set memory limit
    max_limit: int = min(cfg['resource'] * (1024 ** 3), sys.maxsize)
    resource.setrlimit(resource.RLIMIT_AS, (max_limit, resource.RLIM_INFINITY))

    START_TIME: float = time.time()

    # Run scan to estimate period range; auto-select ksize unless --no-auto
    if not cfg.get("no_auto"): # TODO
        logger.info("Auto-selecting ksize using scan")
        cfg = _select_ksize_by_scan_coverage(cfg)
    else:
        logger.info("Running scan for period range estimation (--no-auto: ksize not auto-selected)")
        user_ksize = cfg.get("ksize")
        cfg = _select_ksize_by_scan_coverage(cfg)
        if user_ksize is not None:
            cfg["ksize"] = user_ksize

    # read data
    if not Path(cfg['input']).exists():
        raise FileNotFoundError(cfg['input'])
    with open(cfg['input'], 'r') as fi:
        seq_records: list[SeqIO.SeqRecord] = list(SeqIO.parse(fi, "fasta"))
    logger.info(f"Finished reading fasta file: {cfg['input']}")

    max_seq_len: int = max(len(record.seq) for record in seq_records)
    logger.info(f"Whether to align reverse strand: {cfg['reverse']}")
    if max_seq_len > 5000 and not cfg["reverse"]:
        logger.warning(f"The maximum length of input sequences is {max_seq_len}. If you want to detect potential inversions, please add '--reverse' parameter.")

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
            "copy_number": 0.0,
            "source": "database"
        })
    motif_catalog: pl.DataFrame = pl.DataFrame(rows) # columns: motif, ref_seq, name, source
    logger.info("Finished loading reference motif set")

    # preprocess sequences (remove invalid characters and split into windows)
    SEQNAME2LEN: dict[str, int] = {record.name: len(record.seq) for record in seq_records}
    SEQNAME2SEQ: dict[str, str] = {}
    coordinates_raw2valid: dict[str, np.ndarray] = {}
    coordinates_valid2raw: dict[str, np.ndarray] = {}
    have_invalid: bool = False
    all_sample_kmer_counts: list[dict[int, int]] = []
    with Pool(processes=THREADS) as pool:
        decompose_tasks: list[tuple[str, int, int, np.ndarray]] = []
        for result in tqdm(
            pool.imap(preprocess_sequence, seq_records, chunksize=1),
            total=len(seq_records),
            desc="Preprocessing"
        ):
            seq_name: str = result["seq_name"]
            SEQNAME2SEQ[seq_name] = result["sequence"]
            coordinates_raw2valid[seq_name] = result["coordinates_raw2valid"]
            coordinates_valid2raw[seq_name] = result["coordinates_valid2raw"]
            have_invalid |= result["have_invalid"]
            decompose_tasks.extend(result["tasks"])
            all_sample_kmer_counts.append(result["kmer_counts"])

    # Global k-mer filtering: merge counts across all samples and retain kmers that pass the global threshold
    global_kmer_counts: dict[int, int] = {}
    for kc in all_sample_kmer_counts:
        for k, v in kc.items():
            global_kmer_counts[k] = global_kmer_counts.get(k, 0) + v

    if global_kmer_counts:
        global_max_count: int = max(global_kmer_counts.values())
        global_min_count: int = max(cfg["kmin"], int(global_max_count * cfg["kratio"]))
        global_kmer_set: set[int] = {k for k, v in global_kmer_counts.items() if v >= global_min_count}
        cfg["global_kmer_set"] = global_kmer_set
        logger.info(
            f"Global k-mer filtering: {len(global_kmer_counts)} distinct kmers, "
            f"retained {len(global_kmer_set)} (threshold >= {global_min_count})"
        )
    else:
        cfg["global_kmer_set"] = set()
        logger.warning("No kmers found in any sample during global census.")

    period_range: tuple[int, int] = CFG.get('period_range', (1, 1000))
    logger.info(f"Range of candidate motif length: {period_range[0]}-{period_range[1]} bp")

    # de novo get motifs
    decompose_results: list['Decompose'] = []
    if not cfg.get("no_denovo", False):
        motif_catalog_list: list[pl.DataFrame] = [motif_catalog]
        with Pool(processes=THREADS) as pool:
            for result in tqdm(
                pool.imap(decompose_sequence, decompose_tasks, chunksize=1),
                total=len(decompose_tasks),
                desc="Decomposing"
            ):
                decompose_results.append(result)
                motif_catalog_list.append(result.motif_df)
        motif_catalog_list = [
            x for x in motif_catalog_list
            if x is not None
        ]
        motif_catalog: pl.DataFrames = pl.concat(motif_catalog_list)
        del motif_catalog_list
    else:
        for decompose_task in tqdm(
            decompose_tasks, 
            total=len(decompose_tasks), 
            desc="Skip decomposing"
        ):
            decompose_results.append(do_not_decompose_sequence(decompose_task))

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

    # Recompute copy_number as the exact occurrence count across all input
    # sequences. This replaces the De Bruijn graph edge-weight estimate with the
    # true exact-match copy number to highlight the real mutation conbination.
    candidate_motifs = motif_catalog_denovo["motif"].to_list()

    # Pad sequences with the consensus motif before counting so that motifs
    # crossing the original sequence boundaries are not missed. 
    padding_motif = CFG.get("padding_motif")
    if padding_motif:
        encoded_padding = encode_seq_to_array(padding_motif)
        ksize = CFG["ksize"]
        counting_sequences: list[str] = []
        for seq in SEQNAME2SEQ.values():
            encoded_seq = encode_seq_to_array(seq)
            pad_left, pad_right = _make_adaptive_padding(
                encoded_seq, encoded_padding, ksize
            )
            counting_sequences.append(
                decode_array_to_seq(np.concatenate([pad_left, encoded_seq, pad_right]))
            )
    else:
        counting_sequences = list(SEQNAME2SEQ.values())

    exact_counts = _count_exact_motif_occurrences(
        counting_sequences,
        candidate_motifs,
        include_rc=cfg["reverse"],
    )
    motif_catalog_denovo = motif_catalog_denovo.with_columns(
        pl.col("motif")
        .replace_strict(exact_counts, default=0)
        .cast(pl.Float64)
        .alias("copy_number")
    )
    motif_catalog_denovo = motif_catalog_denovo.filter(pl.col("copy_number") > 0.0)

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
        motif_catalog = motif_catalog.sort("copy_number", descending=True)
        motif_catalog = motif_catalog.with_columns(
            pl.col("motif").str.len_chars().alias("motif_length")
        )
        motif_catalog.write_csv(f"{JOB_DIR}/motif_catalog.tsv", separator="\t", null_value=".")
        print(motif_catalog)

    # update candidate motif set for annotation
    candidate_motifs: list[str] = motif_catalog["motif"].to_list()
    if len(candidate_motifs) == 0:
        raise RuntimeError("No motif detected. Please check the input. You can add '--no-denovo' to use no-denovo mode. Exit.")
    for seg in decompose_results:
        seg.motif_list = candidate_motifs

    # annoate sequence
    dp_tasks: list[tuple(str, str, pl.DataFrame)] = []
    tmp_df_list: list[pl.DataFrame] = []
    cur_seq_name: str = None

    for seg in tqdm(decompose_results, desc="Annotating"):
        seg: Decompose = annotate_sequence(seg, cfg["reverse"])
        seq_name: str = "_".join(seg.name.split("_")[:-1])
        if seq_name != cur_seq_name:
            if len(tmp_df_list) != 0:
                merged_df = pl.concat(tmp_df_list)
                dp_tasks.append((cur_seq_name, SEQNAME2SEQ[cur_seq_name], merged_df))
                tmp_df_list = []
                if cfg.get("debug", False):
                    merged_df.write_csv(f"{JOB_DIR}/annotation/{cur_seq_name}.tsv", separator="\t", null_value=".")
            cur_seq_name = seq_name
        tmp_df_list.append(seg.anno_df)
    if len(tmp_df_list) != 0:
        merged_df = pl.concat(tmp_df_list)
        dp_tasks.append((seq_name, SEQNAME2SEQ[seq_name], merged_df))
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
        pl.col("chrom").replace_strict(SEQNAME2LEN).alias("length")
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
    (
        anno_df
        .select(["chrom", "length", "start", "end", "motif", "orientation", "sequence", "score", "cigar"])
        .write_csv(f"{cfg['prefix']}.annotation.tsv", separator="\t", null_value=".")
    )
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
    concise_df_low_score = concise_df.filter(pl.col("score") <=0)
    if concise_df_low_score.shape[0] > 0:
        logger.warning(f"{concise_df_low_score.shape[0]} sequences have non-positive total scores, which may indicate low confidence. Consider adjusting the scoring parameters or checking the input sequences. Here are the sequences with non-positive scores:\n{concise_df_low_score.select(['chrom', 'score'])}")
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

    # generate raw annotation files if needed
    if cfg.get("use_raw", False):
        raw_anno_df, raw_concise_df, raw_motif_df, raw_dist_df = make_raw(
            anno_df, concise_df, motif_df, dist_df,
            match_score = MATCH_SCORE,
            mismatch_penalty = MISMATCH_PENALTY,
            gap_open_penalty = GAP_OPEN_PENALTY,
            gap_extend_penalty = GAP_EXTEND_PENALTY,
        )
        # save files
        raw_anno_df.write_csv(f"{cfg['prefix']}_raw.annotation.tsv", separator="\t", null_value=".")
        raw_concise_df.write_csv(f"{cfg['prefix']}_raw.concise.tsv", separator="\t", null_value=".")
        raw_motif_df.write_csv(f"{cfg['prefix']}_raw.motif.tsv", separator="\t", null_value=".")
        raw_dist_df.write_csv(f"{cfg['prefix']}_raw.distance.tsv", separator="\t", null_value=".")
        logger.info(
            f"Wrote {cfg['prefix']}_raw.annotation.tsv, {cfg['prefix']}_raw.concise.tsv, "
            f"{cfg['prefix']}_raw.motif.tsv, {cfg['prefix']}_raw.distance.tsv"
        )
        # generate raw h5ad file
        if not cfg.get("skip_h5ad", False):
            import vampire as vp
            adata = vp.anno.pp.read_anno(f"{cfg['prefix']}_raw.annotation.tsv")
            adata.write(f"{cfg['prefix']}_raw.h5ad")
        logger.info(f"Wrote {cfg['prefix']}_raw.h5ad")

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
        web_summary_dst = f"{cfg['prefix']}.web_summary.html"
        shutil.copy2(web_summary_src, web_summary_dst)
        logger.info(f"Generated web summary: {web_summary_dst}")

    logger.info("Bye.")

    # copy log file
    log_path = Path(JOB_DIR) / "log.log"
    if log_path.exists():
        shutil.copy2(f"{JOB_DIR}/log.log", f"{cfg['prefix']}.log")
    else:
        logger.warning(f"Log file not found, skip copying: {log_path}")
