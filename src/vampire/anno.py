import json
import sys, os
import resource
import subprocess
from itertools import compress
from importlib.resources import files
from copy import deepcopy
import time
import sys

import pandas as pd
import edlib
from pybktree import BKTree
import numpy as np
import Levenshtein
from Bio import SeqIO   # I/O processing
from multiprocessing import Pool    # multi-thread
from rich.console import Console

# Self-defined class
from vampire.decomposeString import Decompose       # Decompose and annotate sequences
from vampire.estimateParameters import Estimate    # Automatically find proper parameters

def find_N(task):
    pid, total_task, sequence, args_dict = task
    N_coordinates = []  # List to store the start and end indices of 'N' regions
    start = None  # Variable to track the start of an 'N' region

    # Convert the sequence to a numpy array for efficient processing
    sequence = np.array(list(sequence))
    n_indices = np.where(sequence == 'N')[0]
    
    # If no 'N' characters are found, return an empty DataFrame
    if len(n_indices) == 0:
        log_str = f'Process N Complete: {pid}/{total_task}' if not args_dict['Output Options']['quiet'] else None
        return pd.DataFrame(columns=['start', 'end']), log_str
    
    # Iterate through the indices of 'N' characters to find continuous regions
    for i in range(len(n_indices)):
        if start is None:
            start = n_indices[i]  # Mark the start of a new 'N' region
        
        # If the next 'N' is not consecutive or it's the last 'N', mark the end of the region
        if i == len(n_indices) - 1 or n_indices[i+1] != n_indices[i] + 1:
            end = n_indices[i] + 1  # End index is exclusive
            N_coordinates.append((start, end))  # Add the region to the list
            start = None  # Reset the start for the next region
    
    # Convert the list of 'N' regions to a DataFrame
    N_coordinate_df = pd.DataFrame(N_coordinates, columns=['start', 'end'])
    
    # Adjust the coordinates based on the task ID and window size
    N_coordinate_df[['start', 'end']] += (pid - 1) * args_dict['General Options']['window_length']
    
    log_str = f'Process N Complete: {pid}/{total_task}' if not args_dict['Output Options']['quiet'] else None

    return N_coordinate_df, log_str

def decompose_sequence(task):
    seq_name, pid, total_task, args_dict, sequence = task
    seg = Decompose(sequence, args_dict['Decomposition Options']['ksize'], args_dict['Decomposition Options'], args_dict['Annotation Options'])
    if not args_dict['Decomposition Options']['no_denovo']:
        seg.count_kmers()
        seg.build_dbg(args_dict['I/O Options']['prefix'], seq_name, pid)
        seg.find_motif()
    log_str = f'Decomposition Complete: {pid}/{total_task}' if not args_dict['Output Options']['quiet'] else None
    return ( seq_name, pid, total_task, args_dict, seg ), log_str

def single_motif_annotation(task):
    seq_name, pid, total_task, args_dict, seg = task
    seg.annotate_with_motif()
    # polish 5' end annotation
    ###if pid == 1:
    ###    seg.annotate_polish_head()
    df = seg.annotation
    df[['start', 'end']] += (pid - 1) * (args_dict['General Options']['window_length'] - args_dict['General Options']['overlap_length'])
    df = df.sort_values(by='end').reset_index(drop=True)
    df.to_csv(f"{args_dict['I/O Options']['prefix']}_temp/{seq_name}_anno_{pid}.csv", index = False)
    log_str = f'Single Motif Annotation Complete: {pid}/{total_task}' if not args_dict['Output Options']['quiet'] else None
    return ( seq_name, pid, total_task, args_dict, seg ), log_str

def merge_motifs(task):
    m1, m2, args_dict = task
    m1 = m1.dropna().reset_index(drop=True)
    m2 = m2.dropna().reset_index(drop=True)
    if m1.empty:
        return m2
    if m2.empty:
        return m1

    m1['canonical'] = m1['motif'].apply(canonical_form)
    m2['canonical'] = m2['motif'].apply(canonical_form)

    canonical_to_index = {}
    for idx, row in m1.iterrows():
        canonical = row['canonical']
        canonical_to_index[canonical] = idx

    new_rows = []

    for idx, row in m2.iterrows():
        # check if it is duplicated
        canonical = row['canonical']
        if canonical in canonical_to_index:
            same_idx = canonical_to_index[canonical]
            m1.loc[same_idx, 'value'] += row['value']
        else:
            new_rows.append(row)
            canonical_to_index[canonical] = len(m1) + len(new_rows) - 1

    if new_rows:
        m1 = pd.concat([m1, pd.DataFrame(new_rows)], ignore_index=True)

    m1 = m1.nlargest(args_dict['Decomposition Options']['motifnum'], 'value').reset_index(drop=True)
    return m1

def get_distacne_and_cigar(ref, query):
    matches = edlib.align(query, ref, mode = "NW", task = "path")
    return matches['editDistance'], matches['cigar']

'''def get_distacne_wo_cigar(ref, query):
    matches = edlib.align(query, ref, mode = "NW", task = "distance")
    return matches['editDistance'], "NA"'''

def rc(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N':'N'}
    return ''.join(complement[base] for base in reversed(seq))

def calculate_distance(ref, query_list, motif2id):
    df_list = []
    for query in query_list:
        if len(ref) >= len(query): # keep ref <= query
            tmp_ref, tmp_query = query, ref
        else:
            tmp_ref, tmp_query = ref, query
        
        rep = len(tmp_query) // len(tmp_ref)
        extended_ref = tmp_ref * rep
        extended_ref_rc = rc(tmp_ref) * rep

        min_dist1, min_dist2 = len(extended_ref), len(extended_ref_rc)
        matches_ref = edlib.align(tmp_query, extended_ref, mode="NW", task="distance")
        min_dist1 = min(min_dist1, matches_ref['editDistance'])
        matches_ref_rc = edlib.align(tmp_query, extended_ref_rc, mode="NW", task="distance")
        min_dist2 = min(min_dist2, matches_ref_rc['editDistance'])

        df_list.append([motif2id[ref], motif2id[query], min_dist1, False])
        df_list.append([motif2id[ref], motif2id[query], min_dist2, True])
    return df_list

def canonical_form(motif):
    """Booth's algorithm to find minimal rotation in O(n) time."""
    s = motif + motif
    n = len(s)
    f = [-1] * n
    k = 0
    for j in range(1, n):
        i = f[j - k - 1]
        while i != -1 and s[j] != s[k + i + 1]:
            if s[j] < s[k + i + 1]:
                k = j - i - 1
            i = f[i]
        if s[j] != s[k + i + 1]:
            if s[j] < s[k]:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return s[k:k + len(motif)]

def rotate_strings(s):
    n = len(s)
    return [s[i:] + s[:i] for i in range(n)]

def get_args_json(args_dict):
    return json.dumps(args_dict, indent=4)

def build_args_dict(args):
    return {
        "I/O Options": {
            "input": args.input,
            "prefix": args.prefix
        },
        "General Options": {
            "thread": args.thread,
            "AUTO": args.AUTO,
            "debug": args.debug,
            "window_length": args.window_length,
            "overlap_length": args.overlap_length,
            "resource": args.resource
        },
        "Decomposition Options": {
            "ksize": args.ksize,
            "motif": args.motif,
            "motifnum": args.motifnum,
            "abud_threshold": args.abud_threshold,
            "abud_min": args.abud_min,
            "plot": args.plot,
            "no_denovo": args.no_denovo
        },
        "Annotation Options": {
            "force": True if args.no_denovo else args.force,
            "annotation_dist_ratio": args.annotation_dist_ratio,
            "finding_dist_ratio": args.finding_dist_ratio,
            "match_score": args.match_score,
            "lendif_penalty": args.lendif_penalty,
            "gap_penalty": args.gap_penalty,
            "distance_penalty": args.distance_penalty,
            "perfect_bonus": args.perfect_bonus
        },
        "Output Options": {
            "quiet": args.quiet,
            "score": args.score
        }
    }


# anno.py
def run_anno(args, parser):

    # ------------------------------------------------------------------------
    # config and acquire parameters
    # ------------------------------------------------------------------------
    # load config json file
    args_dict = build_args_dict(args)

    # estimate parameters if set True
    if args.AUTO:
        console.print(f'### Estimate Parameters ###', style="yellow")
        sample_rate = 0.01
        sampled_window_length = 5000
        ksize_list = [3, 5, 9, 11, 13, 15, 31, 41, 71]
        min_length, max_length = 10e3, 50e3

        est = Estimate(args_dict['I/O Options']['input'], sample_rate, ksize_list, sampled_window_length)
        if est.likely_length is None:
            raise ValueError("Cannot detect repeat in the sequence!")
        args_dict['Decomposition Options']['ksize'] = est.get_k()
        del est

    # save parameter settings in JSON format
    args_json = get_args_json(args_dict)
    with open(f"{args_dict['I/O Options']['prefix']}.setting.json", 'w') as setting:
        setting.write(args_json)

    # ------------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------------

    # Create the output directory
    os.makedirs(f"{args_dict['I/O Options']['prefix']}_temp", exist_ok=True)

    # Calculate the step size
    step_length = args_dict['General Options']['window_length'] - args_dict['General Options']['overlap_length']

    # create Console object
    console = Console()

    # set memory limit
    max_limit = min(args_dict['General Options']['resource'] * (1024 ** 3), sys.maxsize)
    resource.setrlimit(resource.RLIMIT_AS, (max_limit, resource.RLIM_INFINITY))

    # ------------------------------------------------------------------------
    # preprocess: Read data and reference motif set, construct BK tree
    # ------------------------------------------------------------------------

    if args_dict['General Options']['debug']:
        start_time = time.time()

    with open(args_dict['I/O Options']['input'], 'r') as handle:
        seq_records = list(SeqIO.parse(handle, "fasta"))
    
    if args_dict['Decomposition Options']['motif'] == 'base':
        db_path = files("vampire.resources").joinpath("refMotif.fa")
    else:
        db_path = args_dict['Decomposition Options']['motif']
    with open(db_path, 'r') as handle:
        motif_records = list(SeqIO.parse(handle, "fasta"))

    ref_motif2name = dict()
    tree = BKTree(Levenshtein.distance)  # use Levenshtein distance
    for record in motif_records:
        motif_name = record.name
        motif = str(record.seq.upper()) # convert to upper case
        tree.add(motif)
        ref_motif2name[motif] = motif_name
        # add inverted motif
        motif_rc = rc(motif)
        tree.add(motif_rc)
        ref_motif2name[motif_rc] = motif_name + '(rc)'

    if args_dict['General Options']['debug']:
        end_time = time.time()
        console.print(f'read data and construct BK tree: {round(end_time - start_time, 2)} s', style="green")

    # ------------------------------------------------------------------------
    # Start processing sequences one by one
    # ------------------------------------------------------------------------
    annotation_list = []
    motif2ref_motif = dict()

    for record in seq_records:
        seq_name = record.name
        console.print(f'### Start Processing [{seq_name}] ###', style="yellow")

        # ------------------------------------------------------------------------
        # deal with N characters
        # ------------------------------------------------------------------------
        if args_dict['General Options']['debug']:
            start_time = time.time()
        
        seq = str(record.seq.upper())

        seqLenwN = len(seq)
        window_length = args_dict['General Options']['window_length']
        overlap_length = args_dict['General Options']['overlap_length']
        step_length = window_length - overlap_length

        total_task = max(1, (seqLenwN - window_length) // step_length + 1)
        
        tasks = [(cur + 1, total_task, seq[cur * step_length : (cur + 1) * step_length], args_dict) for cur in range(total_task)]

        results = []
        with Pool(processes = args_dict['General Options']['thread']) as pool:
            for result, log_str in pool.imap_unordered(find_N, tasks):
                if log_str:
                    console.print(log_str)
                results.append(result)

        N_coordinate = pd.concat(results, ignore_index=True)
        N_coordinate = N_coordinate.sort_values(by=['start']).reset_index(drop=True)


        if N_coordinate.shape[0] > 0:
            N_coordinate_new = pd.DataFrame(columns=['start', 'end'])

            start = N_coordinate.loc[0, 'start']
            end = N_coordinate.loc[0, 'end']
            for i in range(1, N_coordinate.shape[0]):
                if N_coordinate.loc[i, 'start'] != end:
                    N_coordinate_new = pd.concat([N_coordinate_new, pd.DataFrame({'start': [start], 'end': [end]})], ignore_index=True)
                    start = N_coordinate.loc[i, 'start']
                end = N_coordinate.loc[i, 'end']

            # add the last region
            N_coordinate_new = pd.concat([N_coordinate_new, pd.DataFrame({'start': [start], 'end': [end]})], ignore_index=True)

            # solve with 'old_index' and 'new_index'
            N_coordinate_new['old_index'] = N_coordinate_new['end']
            N_coordinate_new['new_index'] = N_coordinate_new['end']

            # calculate new index
            l = 0
            for i in range(N_coordinate_new.shape[0]):
                l += N_coordinate_new.loc[i,'end'] - N_coordinate_new.loc[i,'start']
                N_coordinate_new.loc[i,'new_index'] -= l

        if args_dict['General Options']['debug']:
            end_time = time.time()
            print(f"calculate coordinate transformation: {round(end_time - start_time, 2)} s")

        # ------------------------------------------------------------------------
        # decompose
        # ------------------------------------------------------------------------
        mask = [base != "N" for base in seq]
        filter_seq = ''.join(compress(seq, mask))
        seqLenwoN = len(filter_seq)
        total_task = max(1, (seqLenwoN - overlap_length) // step_length + 1)

        # ------------------------------------------------------------------------
        # decompose
        # ------------------------------------------------------------------------
        if args_dict['General Options']['debug']:
            start_time = time.time()
        
        cur = 0
        tasks = []
        while cur < total_task:
            tasks.append(tuple([seq_name,
                                cur + 1, 
                                total_task, 
                                args_dict,
                                filter_seq[cur*step_length : cur*step_length + window_length]
                                ]))
            cur += 1
        with Pool(processes = args_dict['General Options']['thread']) as pool:
            results = []
            for result, log_str in pool.imap_unordered(decompose_sequence, tasks):
                if log_str:
                    console.print(log_str)
                results.append(result)
            
            pool.close()    # close the pool and don't receive any new tasks
            pool.join()     # wait for all the tasks to complete
            results = list(results)
        
        # ------------------------------------------------------------------------
        # Merge and decide the representive of motifs
        # ------------------------------------------------------------------------
        if not args_dict['Decomposition Options']['no_denovo']:
            motif_df_list = [result.motifs for _, _, _, _, result in results]
            
            cot = 1
            with Pool(processes=args_dict['General Options']['thread']) as pool:
                cot = 1
                while len(motif_df_list) > 1:
                    num = len(motif_df_list)
                    tasks = [tuple([motif_df_list[i*2], motif_df_list[i*2+1], args_dict]) for i in range(num // 2)]
                    if num % 2 == 1:
                        remaining = motif_df_list[num-1]
                    motif_df_list = list(pool.imap_unordered(merge_motifs, tasks))
                    print(f'cycle {cot}...')
                    cot += 1
                    if num % 2 == 1:
                        motif_df_list.append(remaining)

            motifs_df = motif_df_list[0]  # ['motif', 'ref_motif', 'value']
            motifs_df['canonical'] = motifs_df['motif'].apply(canonical_form)
            # merge duplicates in single dataframe
            selected_rows = [0]
            for i in range(1, len(motifs_df)):
                is_skip = False
                for j in range(i):
                    if motifs_df.loc[i, 'canonical'] == motifs_df.loc[j, 'canonical']:
                        motifs_df.loc[i, 'value'] += motifs_df.loc[j, 'value']
                        is_skip = True
                        break
                    if motifs_df.loc[i, 'canonical'] == canonical_form(rc(motifs_df.loc[j, 'motif'])):
                        motifs_df.loc[i, 'value'] += motifs_df.loc[j, 'value']
                        is_skip = True
                        break
                if not is_skip:
                    selected_rows.append(i)

            if motifs_df.shape[0] == 0:
                continue
            motifs_df = motifs_df.loc[selected_rows, :]
            motifs_df = motifs_df.sort_values(by='value', ascending=False).reset_index(drop=True)
            motifs_df = motifs_df.loc[: args_dict['Decomposition Options']['motifnum'] - 1, :]

            console.print(f'Number of identified motif = {motifs_df.shape[0]}', style="white")

            # ------------------------------------------------------------------------
            # polish and refine
            # ------------------------------------------------------------------------
            # polish motif
                # if in reference motif set, adjust to the reference form
                # if reverse complementary sequence is in the reference motif set, invert and adjust form
                # if not in the reference database, find the best form based on its start, 
                # and add into database for other mutant motifs adjustment 
            most_common_motif = motifs_df.loc[0, 'motif']

            for idx in range(motifs_df.shape[0]):
                # try to search in DB
                min_distance = 1e6
                motif = motifs_df.loc[idx, 'motif']
                ref_motif = 'UNKNOWN'
                motif_forms = rotate_strings(motif)  # get all format
                max_distance = int(args_dict['Annotation Options']['finding_dist_ratio'] * len(motif))
                for form in motif_forms:
                    matches = tree.find(form, max_distance)  # find matched motif
                    for dist, ref in matches:
                        if dist < min_distance:
                            motif = rc(form) if '(rc)' in ref_motif2name[ref] else form
                            min_distance = dist
                            ref_motif = rc(ref) if '(rc)' in ref_motif2name[ref] else ref
                    
                # cut the dimer or more
                for form in motif_forms:
                    is_cut = False
                    while True:
                        cur_is_cut = False
                        for ref_motif_2 in ref_motif2name.keys():

                            if len(ref_motif_2) == 1:
                                continue

                            if len(ref_motif_2) >= len(form):
                                continue

                            ratio = len(form) / len(ref_motif_2)

                            if ratio > 3:
                                continue

                            min_ratio = min( abs(ratio - int(ratio)), abs(ratio - int(ratio) - 1) )
                            if min_ratio >= 0.2:
                                continue

                            if form.startswith(ref_motif_2):
                                before = edlib.align(form, ref_motif_2, mode='NW')['editDistance']
                                after = edlib.align(form[len(ref_motif_2):], ref_motif_2, mode='NW')['editDistance']
                                if after < before:
                                    print(f"cut: {form} -> {form[len(ref_motif_2):]}\n\tdistance: {before} -> {after}")
                                    form = form[len(ref_motif_2):]
                                    min_distance_2 = before
                                    motif = motif
                                    ref_motif = ref_motif_2
                                    is_cut = True
                                    cur_is_cut = True
                                    break
                        if not cur_is_cut:
                            break
                    if is_cut:
                        motifs_df.loc[idx, 'motif'] = form
                        motifs_df.loc[idx, 'ref_motif'] = form
                        if form not in ref_motif2name.keys():
                            # add into bktree
                            tree.add(form)
                            ref_motif2name[form] = 'UNKNOWN'
                            motif_rc = rc(motif)
                            tree.add(motif_rc)
                            ref_motif2name[motif_rc] = 'UNKNOWN(rc)'
                        break
                if is_cut:
                    continue
                # searched successfully
                if min_distance < 1e6:
                    motifs_df.loc[idx, 'motif'] = motif
                    motifs_df.loc[idx, 'ref_motif'] = ref_motif
                    motif2ref_motif[motif] = ref_motif
                    continue
                
                # fail to search
                min_index = seqLenwoN + 1e6
                for form in motif_forms:
                    index = filter_seq.find(form)
                    if index < min_index:
                        min_index = index
                        motif = form
                motifs_df.loc[idx, 'motif'] = motif
                # add into reference (temp)
                ### print(f'added {motif}')
                tree.add(motif)
                ref_motif2name[motif] = 'UNKNOWN'
                motif_rc = rc(motif)
                tree.add(motif_rc)
                ref_motif2name[motif_rc] = 'UNKNOWN(rc)'
        else:
            motifs_df = pd.DataFrame(columns=['motif', 'ref_motif', 'value'])
            most_common_motif = motif_records[0].seq.upper()
        
        # add reference motif
        ref_motif2name['UNKNOWN'] = 'UNKNOWN'
        if args_dict['Annotation Options']['force']:
            tmp = []
            for record in motif_records:
                motif_name = record.name
                motif = str(record.seq.upper())
                tmp.append([motif, motif, 0])
            added_motif = pd.DataFrame(tmp, columns=['motif', 'ref_motif', 'value'])
            motifs_df = pd.concat([motifs_df, added_motif], ignore_index = True)

        for i in range(motifs_df.shape[0]):
            motif2ref_motif[motifs_df.loc[i, 'motif']] = motifs_df.loc[i, 'ref_motif']


        ref_motif2name['UNKNOWN'] = 'UNKNOWN'

        nondup = motifs_df['motif'].to_list()
        nondup = list(set(nondup))

        ###print(nondup)

        for _, _, _, _, result in results:
            result.motifs_list = nondup

        if args_dict['General Options']['debug']:
            end_time = time.time()
            print(f"get motif: {round(end_time - start_time, 2)} s")


        # ------------------------------------------------------------------------
        # annotate
        # ------------------------------------------------------------------------
        if args_dict['General Options']['debug']:
            start_time = time.time()
        
        cur = 0
        tasks = results
        results = []
        with Pool(processes = args_dict['General Options']['thread']) as pool:  
            for result, log_str in pool.imap_unordered(single_motif_annotation, tasks):
                if log_str:
                    console.print(log_str)
                results.append(result)
            del tasks
            pool.close()
            pool.join()
                   
        del results
        
        if args_dict['General Options']['debug']:
            end_time = time.time()
            print(f"annotate motif: {round(end_time - start_time, 2)} s")

        if args_dict['General Options']['debug']:
            start_time = time.time()

        # ------------------------------------------------------------------------
        # dynamic programming
        # ------------------------------------------------------------------------

        # parameters
        match_score = args_dict['Annotation Options']['match_score']
        mapped_len_dif_penalty = args_dict['Annotation Options']['lendif_penalty'] # hope mapped sequence is as long as the motif
        gap_penalty = args_dict['Annotation Options']['gap_penalty']
        distance_penalty = args_dict['Annotation Options']['distance_penalty']
        perfect_bonus = args_dict['Annotation Options']['perfect_bonus']

        length = len(filter_seq)
        # Initialize dp and pre arrays
        dp = np.zeros(length + 1, dtype=np.float64)
        pre = np.full((length + 1, 4), None)  # (pre_i, motif_id, motif, dir)

        idx = 0
        current_pid = 1
        df = pd.read_csv(f"{args_dict['I/O Options']['prefix']}_temp/{seq_name}_anno_1.csv")
        anno_start = df['start'].values
        anno_end = df['end'].values
        anno_motif = df['motif'].values
        anno_distance = df['distance'].values
        anno_dir = df['dir'].values

        # Dynamic programming loop
        for i in range(1, length + 1):
            # read annotation data
            pid = (i - overlap_length) // step_length + 1
            if pid > current_pid:
                current_pid = pid
                idx = 0
                df = pd.read_csv(f"{args_dict['I/O Options']['prefix']}_temp/{seq_name}_anno_{pid}.csv")
                anno_start = df['start'].values
                anno_end = df['end'].values
                anno_motif = df['motif'].values
                anno_distance = df['distance'].values
                anno_dir = df['dir'].values

            # Skip one base
            dp[i] = dp[i-1] - gap_penalty
            pre[i] = (i-1, None, None, None)

            # Efficiently iterate over merged_df without repeating checks
            while idx < df.shape[0]:
                if anno_end[idx] > i:
                    break
                if anno_end[idx] < i:
                    idx += 1
                    continue

                pre_i = anno_start[idx]
                motif = anno_motif[idx] ###if anno_dir[idx] == '+' else rc(anno_motif[idx])
                distance = anno_distance[idx]
                bonus = perfect_bonus * ( 2 * np.arctan(len(motif)/10) / np.pi ) * len(motif) if distance == 0 else 0
                if len(motif) >= 2 and canonical_form(motif) == canonical_form(most_common_motif) and bonus > 0:
                    bonus += 2
                score = dp[pre_i] \
                    + len(motif) * match_score \
                    - abs(len(motif) - (i - pre_i)) * mapped_len_dif_penalty \
                    - distance * distance_penalty \
                    + bonus
                if score > dp[i]:
                    dp[i] = score
                    pre[i] = (pre_i, idx, motif, anno_dir[idx])

                idx += 1


            # Print progress every 5000 iterations
            if not args_dict['Output Options']['quiet'] and i % 100e3 == 0:
                console.print(f'DP: {i // 1000} kbp is Done!', style="white")

        console.print('DP complete!', style="green")

        # retrace
        idx = length
        next_coords = [None] * (length + 1)  # to store next coordinates
        next_motif = [None] * (length + 1)   # to store motif references
        next_dir = [None] * (length + 1)

        while idx >= 0:
            if pre[idx][0] is not None:
                if pre[idx][1] is not None:  # matched a motif
                    next_coords[pre[idx][0]] = idx
                    next_motif[pre[idx][0]] = pre[idx][2]
                    next_dir[pre[idx][0]] = pre[idx][3]
                    idx = pre[idx][0]
                else:  # skipped a base
                    next_coords[idx - 1] = idx
                    next_motif[idx - 1] = None
                    next_dir[idx - 1] = None
                    idx -= 1
            else:
                idx -= 1
  
        console.print('Retrace Complete!', style="green")

        if args_dict['General Options']['debug']:
            end_time = time.time()
            print(f"DP and retrace: {round(end_time - start_time, 2)} s")

        if args_dict['General Options']['debug']:
            start_time = time.time()

        console.print(f'### Organizing results ###', style="yellow")

        # output motif annotation file
        annotation_data = []
        idx = 0
        

        while idx < length:
            if pre[idx][0] is None and next_coords[idx] is not None:    # is a start site
                idx2 = idx
                start, end = idx, 0
                cur_motif, cur_dir, rep_num, score, max_score, cigar_string = None, None, 0, 0, 0, ''
                skip_num = 0
                row = None

                while next_coords[idx2] is not None:
                    ############  cycle start  ############
                    '''
                    logic:
                    1. motif aligned
                        - output if have a different motif / dir
                        - add skip cigar string
                        - add score of alignment and cigar string
                        - check if score is max

                    2. motif unaligned
                        - add skip penalty
                        - check if score is lower than 0.98 * max_score
                    '''
                    ###print(idx2)
                    motif = next_motif[idx2]
                    dir = next_dir[idx2]
                    if motif is not None:    # match a motif 
                        if cur_motif != motif or cur_dir != dir:       # split the annotation and init
                            if cur_motif is not None and row is not None:
                                annotation_data.append(row)
                            start, cur_motif, cur_dir, rep_num, score, max_score, cigar_string = idx2, motif, dir, 0, 0, 0, ''
                            skip_num = 0

                        if skip_num:
                            cigar_string += f'{skip_num}N'
                            skip_num = 0

                        rep_num += 1
                        tmp_motif = motif if dir == '+' else rc(motif)
                        distance, cigar = get_distacne_and_cigar(tmp_motif, filter_seq[idx2: next_coords[idx2]])
                        '''if dir == "-":
                            print(distance, cigar)'''
                        if idx2 == 0:
                            distance -= len(motif) - len(filter_seq[idx2: next_coords[idx2]])
                        score += len(cur_motif) * match_score - distance * distance_penalty
                        cigar_string += cigar + '/'
                        if score >= max_score:
                            row = [seq_name, seqLenwN, start, next_coords[idx2], motif, dir, rep_num, score, cigar_string]
                            max_score = score
                    else:               # skip a base
                        if idx2 != 0 and cur_motif is not None:
                            skip_num += 1
                            score -= gap_penalty
                            if score < 0.98 * max_score:
                                ###if row not in annotation_data:
                                annotation_data.append(row)
                                cur_motif, cur_dir, rep_num, score, max_score, cigar_string = None, None, 0, 0, 0, ''
                                skip_num = 0

                
                    idx2 = next_coords[idx2]
                    end = idx2
                    ############  cycle end  ############
                if row is not None:
                    annotation_data.append(row)
            break

        unique_results = {tuple(r) for r in annotation_data}
        annotation_data = [list(r) for r in unique_results]


        for j in range(1, len(annotation_data)):
            if annotation_data[j] == annotation_data[j-1]:
                raise ValueError(f"Duplicate found")

        annotation = pd.DataFrame(annotation_data, columns=['seq','length','start','end','motif', 'dir', 'rep_num','score','CIGAR'])
        annotation = annotation.reset_index(drop=True)
        
        # coordinates transformation with N character
        if N_coordinate.shape[0]:
            for i in range(annotation.shape[0]):
                # edit cigar string
                old_cigar = annotation.loc[i, 'CIGAR']
                new_cigar, cot = '', ''
                idx = 0
                cur = annotation.loc[i, 'start']
                while idx < len(old_cigar):
                    symbol = old_cigar[idx]
                    if symbol == '/':
                        new_cigar += '/'
                    elif symbol in ['=','X','I','N']:
                        length = int(cot)
                        tmp = N_coordinate_new[(N_coordinate_new['new_index'] > cur) & (N_coordinate_new['new_index'] <= cur + length)]
                        if tmp.shape[0]:    # if there is N character in this region
                            
                            for row in tmp.itertuples():
                                length -= row.new_index - cur
                                new_cigar += f'{row.new_index - cur}{symbol}'
                                new_cigar += f'{row.end - row.start}N'
                                cur += row.new_index - cur
                                ### new_cigar += f'{N_coordinate_new['new_index'] - cur}{symbol}'
                            new_cigar += f'{length}{symbol}' if length != 0 else ''
                            cur += length
                        else:
                            new_cigar += f'{length}{symbol}'
                            cur += length
                        cot = ''
                    elif symbol == 'D':
                        new_cigar += f'{cot}{symbol}'
                        cot = ''
                    else:
                        cot += symbol
                    
                    annotation.loc[i, 'CIGAR'] = new_cigar
                    idx += 1
                
                
                # transform start site
                tmp = N_coordinate_new[N_coordinate_new['new_index'] <= int(annotation.loc[i,'start'])]
                if tmp.shape[0]:
                    tmp = tmp.reset_index(drop=True)
                    l = tmp.shape[0] - 1
                    annotation.loc[i,'start'] += tmp.loc[l,'old_index'] - tmp.loc[l,'new_index']
                # transform end site
                tmp = N_coordinate_new[N_coordinate_new['new_index'] <= int(annotation.loc[i,'end'])]
                if tmp.shape[0]:
                    tmp = tmp.reset_index(drop=True)
                    l = tmp.shape[0] - 1
                    annotation.loc[i,'end'] += tmp.loc[l,'old_index'] - tmp.loc[l,'new_index'] 
        annotation_list.append(annotation)
        console.print(f'### Complete [{seq_name}] ###\n', style="yellow")

    # ------------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------------

    merged_annotation = pd.concat(annotation_list, ignore_index=True)
    merged_annotation = merged_annotation[merged_annotation['score'] >= args_dict['Output Options']['score']]
    merged_annotation = merged_annotation.sort_values(by=["seq", "start"]).reset_index(drop=True)
    merged_annotation.to_csv(f"{args_dict['I/O Options']['prefix']}.concise.tsv", sep = '\t', index = False)


    if args_dict['General Options']['debug']:
        end_time = time.time()
        print(f"make concise.tsv: {round(end_time - start_time, 2)} s")

    console.print('output concise.tsv!', style="green")

    if args_dict['General Options']['debug']:
        start_time = time.time()

    if not args_dict['General Options']['debug']:
        subprocess.run(f"rm -r {args_dict['I/O Options']['prefix']}_temp", shell = True)

    # ------------------------------------------------------------------------
    # make file - *.annotation.tsv
    # ------------------------------------------------------------------------
    detail_annotation = []
    for record in seq_records:
        seq_name = record.name
        sequence = str(record.seq.upper())
        seqLen = len(sequence)
        tmp = merged_annotation[merged_annotation['seq'] == seq_name]
        
        for index, row in tmp.iterrows():
            idx = row.start
            cigar_string = row.CIGAR
            dir = row.dir
            j = 0
            start, length, dist, actual_motif, sub_cigar, num = row.start, 0, 0, '', '', ''
            while j < len(cigar_string):
                symbol = cigar_string[j]
                if symbol == '/':
                    detail_annotation.append([seq_name, seqLen, start, start + length, row.motif, dir, dist, actual_motif, sub_cigar])
                    start = start + length
                    length, dist, actual_motif, sub_cigar = 0, 0, '', ''
                elif symbol in ['=','X','I']:
                    actual_motif += sequence[start + length : start + length + int(num)]
                    length += int(num)
                    if symbol != '=':
                        dist += int(num)
                    sub_cigar += f'{num}{symbol}'
                    num = ''
                elif symbol == 'D':
                    sub_cigar += f'{num}D'
                    dist += int(num)
                    num = ''
                elif symbol == 'N':
                    actual_motif += sequence[start + length : start + length + int(num)]
                    length += int(num)
                    sub_cigar += f'{num}N'
                    num = ''
                else:
                    num += symbol
                j += 1
            ### detail_annotation.append([seq_name, seqLen, start, start + length, row.motif, actual_motif, sub_cigar])

    detailed_df = pd.DataFrame(data = detail_annotation, columns = ['seq','length','start','end','motif','dir','distance','actual_motif','CIGAR'])
    detailed_df.to_csv(f"{args_dict['I/O Options']['prefix']}.annotation.tsv", sep = '\t', index = False)

    if args_dict['General Options']['debug']:
        end_time = time.time()
        print(f"make annotation.tsv: {round(end_time - start_time, 2)} s")

    console.print('output annotation.tsv!', style="green")

    if args_dict['General Options']['debug']:
        start_time = time.time()

    # ------------------------------------------------------------------------
    # make file - *.motif.tsv
    # ------------------------------------------------------------------------
    rep_num = []
    motif_group = merged_annotation.groupby('motif')['rep_num'].sum().reset_index()
    motif_df = motif_group[['motif', 'rep_num']].copy()
    motif_df = motif_df.sort_values(by=['rep_num'], ascending = False).reset_index(drop=True)
    motif_df.index.name = 'id'
    motifs_list = motif_df['motif'].to_list()
    tmp = []
    for index, row in motif_df.iterrows():
        tmp.append(ref_motif2name[motif2ref_motif[row.motif]])

    label_col = {'label' : tmp}
    label_col_df = pd.DataFrame(label_col)
    motif_df = pd.concat([motif_df, label_col_df], axis=1)
    motif_df.to_csv(f"{args_dict['I/O Options']['prefix']}.motif.tsv", sep = '\t', index = True, index_label='id')

    motif2id = {row['motif'] : index for index, row in motif_df.iterrows()}
    
    if args_dict['General Options']['debug']:
        end_time = time.time()
        print(f"make motif.tsv: {round(end_time - start_time, 2)} s")

    console.print('output motif.tsv!', style="green")

    if args_dict['General Options']['debug']:
        start_time = time.time()

    # ------------------------------------------------------------------------
    # make file - *.dist.tsv
    # ------------------------------------------------------------------------
    all_df_row = []
    for motif in motifs_list:
        all_df_row.extend(calculate_distance(motif, motifs_list, motif2id))
        

    distance_df = pd.DataFrame(all_df_row, columns = ['ref','query','dist','is_rc'])
    distance_df['sum'] = distance_df['ref'] + distance_df['query']
    distance_df = distance_df[distance_df['ref'] != distance_df['query']]
    distance_df = distance_df.sort_values(by=['dist','sum','ref','query']).reset_index(drop=True)
    distance_df.to_csv(f"{args_dict['I/O Options']['prefix']}.dist.tsv", sep = '\t', index = False, columns = ['ref','query','dist','is_rc'])

    if args_dict['General Options']['debug']:
        end_time = time.time()
        print(f"make dist.tsv: {round(end_time - start_time, 2)} s")

    console.print('output dist.tsv!', style="green")
