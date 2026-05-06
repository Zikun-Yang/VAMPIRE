import argparse
import pandas as pd
import numpy as np
import edlib
from collections import Counter
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
from tqdm import tqdm


def read_file(prefix: str) -> tuple:
    # id, motif, rep_num, label
    motif_path = f'{prefix}.motif.tsv'
    motif = pd.read_table(motif_path, sep = '\t', header = 0)
    # ref, query, dist, is_rc
    dist_path = f'{prefix}.dist.tsv'
    dist = pd.read_table(dist_path, sep = '\t', header = 0)
    # seq, length, start, end, motif, rep_num, score, CIGAR
    concise_path = f'{prefix}.concise.tsv'
    concise = pd.read_table(concise_path, sep = '\t', header = 0)
    # seq, length, start, end, motif, actual_motif, CIGAR
    annotation_path = f'{prefix}.annotation.tsv'
    annotation = pd.read_table(annotation_path, sep = '\t', header = 0)
    return motif, dist, concise, annotation

'''def prepare_annotation(annotation: pd.DataFrame, motif2id: dict) -> pd.DataFrame:
    annotation['motif_id'] = annotation['motif'].map(motif2id)
    return annotation'''

def prepare_dist_matrices(dist: pd.DataFrame) -> tuple:
    # make dist matrix
    df_wide = dist[~dist['is_rc']].sort_values(by = ['ref', 'query']).pivot_table(index='ref', columns='query', values='dist') # avoid duplicate values
    dist_matrix = df_wide.values
    # distance = 0 if ref == query
    np.fill_diagonal(dist_matrix, 0)

    df_wide = dist[dist['is_rc']].sort_values(by = ['ref', 'query']).pivot_table(index='ref', columns='query', values='dist') # avoid duplicate values
    rc_dist_matrix = df_wide.values
    # distance = 0 if ref == query
    np.fill_diagonal(rc_dist_matrix, 0)
    
    return dist_matrix, rc_dist_matrix

def rc(motif: str) -> str:
    reverse_complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join([reverse_complement[base] for base in motif[::-1]])

def get_motifs(task: tuple) -> tuple:
    window, sub_annotation, window_size = task
    if sub_annotation[sub_annotation['window'] == window].shape[0] < 0.5 * window_size:
        return None, (None, None)

    motif = sub_annotation[(sub_annotation['window'] == window)]['actual_motif'].tolist()
    motif_direction = sub_annotation[(sub_annotation['window'] == window)]['dir'].tolist()
    # invert if motif_direction is '-'
    motif = [motif[i] if motif_direction[i] == '+' else rc(motif[i]) for i in range(len(motif))]

    motif_length = sub_annotation[(sub_annotation['window'] == window)]['end'] - sub_annotation[(sub_annotation['window'] == window)]['start']
    motif_length = motif_length.tolist()
    
    seq = ''.join(motif[::-1]) if '+' not in motif_direction else ''.join(motif)

    return seq, (motif, motif_direction)

def get_extend_seq_for_precise(m1: tuple, m2: tuple, m3: tuple) -> tuple:
    m1 = list(m1)
    m2 = list(m2)
    m3 = list(m3)
    if m1[0] is not None and len(m1[0]) >=2:
        l = len(m1[0]) // 2
        m2[0] = m1[0][:l] + m2[0]
        m2[1] = m1[1][:l] + m2[1]
    
    if m3[0] is not None and len(m3[0]) >= 2:
        l = len(m3[0]) // 2
        m2[0] = m2[0] + m3[0][l:]
        m2[1] = m2[1] + m3[1][l:]

    return tuple(m2)

def extend_motifs(task: tuple) -> list:
    window, raw_motifs_window = task
    
    if raw_motifs_window[window][0] is None:
        return None
    else:
        l_notna = True
        r_notna = True
        if window == 0 or raw_motifs_window[window - 1][0] is None:
            l_notna = False
        if window == len(raw_motifs_window) - 1 or raw_motifs_window[window + 1][0] is None:
            r_notna = False

        l = raw_motifs_window[window - 1] if l_notna else (None, None)
        r = raw_motifs_window[window + 1] if r_notna else (None, None)
        result = get_extend_seq_for_precise(l, raw_motifs_window[window], r)
        result = list(result)
        if '+' not in result[1]:
            result[0] = result[0][::-1]
            result[1] = result[1][::-1]
        seq = ''.join(result[0])
        return seq

'''
def get_motifs(task: tuple) -> tuple:
    window, sub_annotation, window_size, precise = task
    if sub_annotation[sub_annotation['window'] == window].shape[0] < 0.5 * window_size:
        return None, None, None
    if precise:
        motif = sub_annotation[(sub_annotation['window'] == window)]['actual_motif'].tolist()
        motif_direction = sub_annotation[(sub_annotation['window'] == window)]['dir'].tolist()
        # invert if motif_direction is '-'
        motif = [motif[i] if motif_direction[i] == '+' else rc(motif[i]) for i in range(len(motif))]
    else:
        motif = sub_annotation[(sub_annotation['window'] == window)]['motif_id'].tolist()
        motif_direction = sub_annotation[(sub_annotation['window'] == window)]['dir'].tolist()
    motif_length = sub_annotation[(sub_annotation['window'] == window)]['end'] - sub_annotation[(sub_annotation['window'] == window)]['start']
    motif_length = motif_length.tolist()
    #motif_distance = sub_annotation[(sub_annotation['window'] == window)]['distance'].tolist()
    return tuple([motif, motif_direction, motif_length])
'''

def invert_annotation(annotation: pd.DataFrame) -> pd.DataFrame:
    sub_annotation_list = []
    for seq, sub_annotation in annotation.groupby('seq'):
        sub_annotation = sub_annotation.copy().sort_index()

        # identity minus strand region
        cur_minus = False
        for index, row in sub_annotation.iterrows():
            if row['dir'] == '-':
                if not cur_minus:
                    start = index
                cur_minus = True
            else:
                if cur_minus:
                    end = index
                    ###sub_annotation.loc[start:end, 'actual_motif'] = sub_annotation.loc[start:end, 'actual_motif'][::-1]
                    sub_annotation.iloc[start:end, sub_annotation.columns.get_loc('actual_motif')] = \
                        sub_annotation.loc[start:end-1, 'actual_motif'].iloc[::-1].apply(rc)
                    sub_annotation.iloc[start:end, sub_annotation.columns.get_loc('dir')] = '+'
                    ###sub_annotation.loc[start:end, 'distance'] = sub_annotation.loc[start:end, 'distance'][::-1]
                    cur_minus = False
        
        if cur_minus:
            sub_annotation.iloc[start:, sub_annotation.columns.get_loc('actual_motif')] = \
                sub_annotation.loc[start:]['actual_motif'].iloc[::-1].apply(rc)
            sub_annotation.iloc[start:, sub_annotation.columns.get_loc('dir')] = '+'
            ###sub_annotation.loc[start:, 'actual_motif'] = sub_annotation.loc[start:, 'actual_motif'][::-1]
            ###sub_annotation.loc[start:, 'distance'] = sub_annotation.loc[start:, 'distance'][::-1]

        sub_annotation_list.append(sub_annotation)

    inverted_annotation = pd.concat(sub_annotation_list)
    inverted_annotation.reset_index(inplace=True, drop = True)

    return inverted_annotation

def remove_overlap(motif1, motif2):
    # Create counters for motif1 and motif2
    counter1 = Counter(motif1)
    counter2 = Counter(motif2)

    # Find common motifs (intersection of keys)
    common_motifs = set(counter1) & set(counter2)

    # Loop through each common motif
    for m in common_motifs:
        # Calculate the overlap based on the minimum count in both lists
        overlap_num = min(counter1[m], counter2[m])
        
        # Subtract the count of the motif from both counters
        counter1[m] -= overlap_num
        counter2[m] -= overlap_num

    # Reconstruct the motif lists after removal
    motif1 = list(counter1.elements())
    motif2 = list(counter2.elements())
    return motif1, motif2


'''
def max_identity(m1, m2, dist_matrix: np.ndarray, max_dist: int):
    motif1, motif1_distance, motif1_length = m1
    motif2, motif2_distance, motif2_length = m2
    
    motif1_length = np.asarray(motif1_length)
    motif2_length = np.asarray(motif2_length)

    motif1, motif2 = remove_overlap(motif1, motif2)

    if len(motif1) == 0 or len(motif2) == 0:
        return 0
    
    new_dist_matrix =  np.zeros((len(motif1), len(motif2)))
    for i in range(len(motif1)):
        for j in range(len(motif2)):
            new_dist_matrix[i, j] = dist_matrix[motif1[i], motif2[j]] # + motif1_distance[i] + motif2_distance[j]

    inf = 1e6

    if len(motif1) > len(motif2):
        dummy_col = np.full((len(motif1), len(motif1) - len(motif2)), inf)
        new_dist_matrix = np.hstack((new_dist_matrix, dummy_col))
    elif len(motif2) > len(motif1):
        dummy_row = np.full((len(motif2) - len(motif1), len(motif2)), inf)
        new_dist_matrix = np.vstack((new_dist_matrix, dummy_row))

    ###print("New Distance Matrix before padding:")
    ###print(new_dist_matrix)

    row_ind, col_ind = linear_sum_assignment(new_dist_matrix)
    min_dist = new_dist_matrix[row_ind, col_ind]

    unmask = min_dist <= max_dist  # get boolean mask
    valid_rows = row_ind[unmask]   # get valid rows
    valid_motif1_length = motif1_length[valid_rows].sum()
    valid_cols = col_ind[unmask]   # get valid cols
    valid_motif2_length = motif2_length[valid_cols].sum()
    valid_min_dist = min_dist[unmask]

    denominator = valid_motif1_length + valid_motif2_length
    if denominator == 0:
        ###print(valid_motif1_length, valid_motif2_length)
        dist_result = np.nan  # 或者设置为某个默认值，比如 1 或 inf
    else:
        dist_result = valid_min_dist.sum() / denominator * 2

    ###dist_result = valid_min_dist.sum() / (valid_motif1_length + valid_motif2_length) * 2

    return float(dist_result)

def calculate_identity_without_order(task: tuple) -> tuple:
    window1, window2, motifs_window, dist_matrix, max_dist = task

    if motifs_window[window1][0] is None or motifs_window[window2][0] is None:
        return None, None, None

    l_notna = True
    r_notna = True
    if window1 == 0 or motifs_window[window1 - 1][0] is None:
        l_notna = False
    if window1 == len(motifs_window) - 1 or motifs_window[window1 + 1][0] is None:
        r_notna = False
    l = motifs_window[window1 - 1][0] if l_notna else []
    r = motifs_window[window1 + 1][0] if r_notna else []
    window1_extend = l + motifs_window[window1][0] + r

    l_notna = True
    r_notna = True
    if window2 == 0 or motifs_window[window2 - 1][0] is None:
        l_notna = False
    if window2 == len(motifs_window) - 1 or motifs_window[window2 + 1][0] is None:
        r_notna = False
    l = motifs_window[window2 - 1][0] if l_notna else []
    r = motifs_window[window2 + 1][0] if r_notna else []
    window2_extend = l + motifs_window[window2][0] + r

    id1 = max_identity(motifs_window[window1], window2_extend, dist_matrix, max_dist)
    id2 = max_identity(motifs_window[window2], window1_extend, dist_matrix, max_dist)

    identity_result = max(id1, id2)

    return window1, window2, identity_result
'''

def process_cigar(cigar: str, min_indel: int, max_indel: int) -> tuple:
    # symbol: =, X, I, D
    match_base = 0
    mismatch_base = 0
    insert_event = 0
    delete_event = 0
    num = ''
    for i in cigar:
        if i not in ['=', 'X', 'I', 'D']:
            num += i
        else:
            if i == '=':
                match_base += int(num)
            elif i == 'X':
                mismatch_base += int(num)
            elif i == 'I':
                if int(num) >= min_indel and int(num) <= max_indel:
                    insert_event += 1
            elif i == 'D':
                if int(num) >= min_indel and int(num) <= max_indel:
                    delete_event += 1
            num = ''

    return match_base, mismatch_base, insert_event, delete_event



def calculate_identity_precisely(task: tuple) -> tuple:
    window1, window2, seq1, seq2, seq1_extend, seq2_extend, min_indel, max_indel = task

    if seq1 is None or seq2 is None:
        return None, None, None
    
    if window1 == window2:
        return window1, window2, 100

    alignment = edlib.align(seq1, seq2_extend, mode='HW', task='path')
    match_base, mismatch_base, insert_event, delete_event = process_cigar(alignment['cigar'], min_indel, max_indel)
    ###print(match_base, (match_base + mismatch_base + insert_event + delete_event))
    id1 = match_base / (match_base + mismatch_base + insert_event + delete_event) * 100

    alignment = edlib.align(seq1_extend, seq2, mode='HW', task='path')
    match_base, mismatch_base, insert_event, delete_event = process_cigar(alignment['cigar'], min_indel, max_indel)
    ###print(match_base, (match_base + mismatch_base + insert_event + delete_event))
    id2 = match_base / (match_base + mismatch_base + insert_event + delete_event) * 100

    return window1, window2, max(id1, id2)

'''
def calculate_identity_with_order(task: tuple) -> tuple:
    window1, window2, motifs_window, dist_matrix, max_dist = task

    # get the motif of the window1 and window2
    motif1, motif1_direction = motifs_window[window1]
    motif2, motif2_direction = motifs_window[window2]

    # get the dist of the window1 and window2
    if len(motif1) == 0 or len(motif2) == 0:
        print(f'window1: {window1}, window2: {window2}, motif1: {motif1}, motif2: {motif2}')
        return np.nan
    
    if len(motif1) == len(motif2):
        total_dist_list = []
        total_dist = 0
        for i in range(len(motif1)):
            total_dist += dist_matrix[motif1[i], motif2[i]]
            # if motif1_direction[i] == motif2_direction[i]:
            #     total_dist += dist_matrix[motif1[i], motif2[i]]
            # else:
            #     total_dist += rc_dist_matrix[motif1[i], motif2[i]]
        total_dist_list.append(total_dist)

        # start from 1
        new_motif2 = motif2[1:] + [motif2[0]]
        new_motif2_direction = motif2_direction[1:] + [motif2_direction[0]]
        for i in range(len(motif1)):
            total_dist += dist_matrix[motif1[i], new_motif2[i]]
            # if motif1_direction[i] == new_motif2_direction[i]:
            #     total_dist += dist_matrix[motif1[i], new_motif2[i]]
            # else:
            #     total_dist += rc_dist_matrix[motif1[i], new_motif2[i]]
        total_dist_list.append(total_dist)
        ###print(total_dist_list)
        total_dist = min(total_dist_list)

    else:
        total_dist_list = []
        if len(motif1) < len(motif2): # swap motif1 and motif2
            motif1, motif2 = motif2, motif1
            motif1_direction, motif2_direction = motif2_direction, motif1_direction
        
        for i in range(len(motif1) - len(motif2) + 1): # match and use the mean value of the dist
            total_dist = 0
            for j in range(len(motif2)):
                total_dist += dist_matrix[motif1[j + i], motif2[j]]
                # if motif1_direction[j + i] == motif2_direction[j]:
                #     total_dist += dist_matrix[motif1[j + i], motif2[j]]
                # else:
                #     total_dist += rc_dist_matrix[motif1[j + i], motif2[j]]
            total_dist_list.append(total_dist)
        total_dist = min(total_dist_list)

    return window1, window2, total_dist
'''

# identity.py
def run_identity(args, parser):
    prefix = args.prefix
    window_size = args.window_size

    # print mode
    print(f"mode: {args.mode}")

    # read files
    motif, dist, concise, annotation = read_file(prefix)
    seq_names = list(concise['seq'].unique())
    motif2id = {row['motif']: int(row['id']) for _, row in motif.iterrows()}
    id2motif = {row['id']: row['motif'] for _, row in motif.iterrows()}

    if args.mode == 'invert':
        print('inverting annotation')
        annotation = invert_annotation(annotation)

    '''
    # compare aff_annotation and annotation on minus strand
    aff_annotation_minus = aff_annotation[aff_annotation['dir'] == '-'].reset_index(drop=True)
    annotation_minus = annotation[annotation['dir'] == '-'].reset_index(drop=True)
    print(aff_annotation_minus.shape[0], annotation_minus.shape[0])

    if aff_annotation_minus.shape[0] == annotation_minus.shape[0]:
        for i in range(aff_annotation_minus.shape[0]):
            if aff_annotation_minus.loc[i, 'actual_motif'] != annotation_minus.loc[i, 'actual_motif']:
                print(i)
    '''
    ###annotation = prepare_annotation(annotation, motif2id)  # Preprocess annotation

    # Make dist matrices
    ###dist_matrix, rc_dist_matrix = prepare_dist_matrices(dist)

    #query_name     query_start     query_end       reference_name  reference_start reference_end   perID_by_events 
    for seq in seq_names:
        sub_annotation = annotation[annotation['seq'] == seq].reset_index(drop=True)
        sub_annotation['window'] = sub_annotation.index // window_size
        window_num = sub_annotation['window'].max() + 1

        window2start = {i: sub_annotation[sub_annotation['window'] == i]['start'].min() for i in range(window_num)}
        window2end = {i: sub_annotation[sub_annotation['window'] == i]['end'].max() for i in range(window_num)}
        window2length = {i: window2end[i] - window2start[i] for i in range(window_num)}

        tasks = [(i, sub_annotation, args.window_size) for i in range(window_num)]
        # get motifs for each window
        with Pool(args.thread) as pool:
            result = list(tqdm(pool.imap(get_motifs, tasks), total=len(tasks), desc='building motifs'))

        motifs_window = [result[i][0] for i in range(len(result))]
        raw_motifs_window = [result[i][1] for i in range(len(result))]

        tasks = [(i, raw_motifs_window) for i in range(window_num)]
        with Pool(args.thread) as pool:
            motifs_extended_window = list(tqdm(pool.imap(extend_motifs, tasks), total=len(tasks), desc='extending motifs'))

        # generate tasks
        tasks = []
        for i in range(window_num):
            for j in range(i, window_num):
                tasks.append((i, j, motifs_window[i], motifs_window[j], motifs_extended_window[i], motifs_extended_window[j], args.min_indel, args.max_indel))
            # calculate identity matrix
        with Pool(args.thread) as pool:
            identity_matrix = list((tqdm(pool.imap(calculate_identity_precisely, tasks), total=len(tasks), desc='calculating identity')))

        ###print(identity_matrix[:5])


        # process the identity matrix
        identity_matrix = pd.DataFrame(identity_matrix, columns = ['window1', 'window2', 'perID_by_events'])
        identity_matrix = identity_matrix.dropna()
        identity_matrix['query_name'] = seq
        identity_matrix['reference_name'] = seq
        identity_matrix['query_start'] = identity_matrix['window1'].map(window2start)
        identity_matrix['query_end'] = identity_matrix['window1'].map(window2end)
        identity_matrix['reference_start'] = identity_matrix['window2'].map(window2start)
        identity_matrix['reference_end'] = identity_matrix['window2'].map(window2end)
        identity_matrix = identity_matrix[['query_name', 'query_start', 'query_end', 'reference_name', 'reference_start', 'reference_end', 'perID_by_events']]

        # save the identity matrix
        filename = f'{args.output}_{seq}.bed'
        with open(filename, 'w') as f:
            f.write('#' + '\t'.join(identity_matrix.columns) + '\n')
            identity_matrix.to_csv(f, sep='\t', index=False, header=False)
