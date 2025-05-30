import edlib
import numpy as np
import pandas as pd
from multiprocessing import Pool

def parse_string(s):
    sources = []
    for source in s.split(","):
        if "+" in source:
            sources.append([source.strip("+"), "+"])
        elif "-" in source:
            sources.append([source.strip("-"), "-"])
        else:
            sources.append([source, None])
    return sources

def parse_and_validate_actions(action_df):
    actions = []
    #seen_targets = defaultdict(set)
    
    for _, row in action_df.iterrows():
        source = row['source']
        action = row['action']
        actions.append((parse_string(source), action))

    # conflict 1: multiple action
    seen = []
    for source, action in actions:
        for s, dir in source:
            if dir is None:
                name = f"{s}+"
                if name in seen:
                    raise ValueError(f"Single objective {name} corresponds to multiple actions.")
                else:
                    seen.append(name)
                name = f"{s}-"
                if name in seen:
                    raise ValueError(f"Single objective {name} corresponds to multiple actions.")
                else:
                    seen.append(name)
            else:
                name = f"{s}{dir}"
                if name in seen:
                    raise ValueError(f"Single objective {name} corresponds to multiple actions.")
                else:
                    seen.append(name)
    
    return actions

def compute_edit_distance(str1, str2):
    alignment = edlib.align(str1, str2, mode = 'HW', task='distance')
    return alignment['editDistance']

def get_cigar(str1, str2):
    alignment = edlib.align(str1, str2, mode = 'NW', task='path')
    return alignment['cigar']

def rc(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N':'N'}
    return ''.join(complement[base] for base in reversed(seq))


def merge_calculate_distance(task):
    index, target_motif, motif, dir = task
    ###print(index)
    min_distance = 1e6
    if dir is None:
        distance = compute_edit_distance(target_motif, motif)
        if distance < min_distance:
            min_distance = distance
            dir = '+'
        distance = compute_edit_distance(target_motif, rc(motif))
        if distance < min_distance:
            min_distance = distance
            dir = '-'
    elif dir == '+':
        distance = compute_edit_distance(target_motif, motif)
        if distance < min_distance:
            min_distance = distance
    elif dir == '-':
        distance = compute_edit_distance(target_motif, rc(motif))
        if distance < min_distance:
            min_distance = distance
    else:
        raise ValueError("unrecognized direction!")
    return (index, target_motif, motif, dir, min_distance)

def merge(annotation, sources, id2motif, args):
    # get index to revise
    index_list = []
    for source, dir in sources:
        motif = id2motif[source]
        if dir is None:
            mask = (annotation['motif'] == motif)
        else:
            mask = (annotation['motif'] == motif) & (annotation['dir'] == dir)
        index_list.extend(annotation[mask].index.tolist())
    index_list = list(set(index_list))
    target_motifs = annotation.loc[index_list,]['actual_motif'].tolist()

    ###print(index_list)

    min_total_distance = None
    best_alternative = None
    for source, dir in sources:
        ###print(f"{source}{dir}")
        alternative = id2motif[source]
        tasks = [(index_list[i], target_motifs[i], alternative, dir) for i in range(len(index_list))]         
        ###print(tasks)
        with Pool(processes=args.thread) as pool:
            results = pool.imap_unordered(merge_calculate_distance, tasks)
            pool.close()
            pool.join()
        results = list(results)
        
        ###print('done!')
        total_distance = sum(map(lambda x: x[4], results))
        if min_total_distance is None or total_distance < min_total_distance:
            min_total_distance = total_distance
            best_alternative = results

    max_distance = len(best_alternative[0][2]) * threshold
    for result in best_alternative:
        index, target_motif, motif, dir, min_distance = result
        if min_distance < max_distance:
            annotation.loc[index, 'motif'] = motif
            annotation.loc[index, 'dir'] = dir
            annotation.loc[index, 'CIGAR'] = get_cigar(target_motif, motif if dir == '+' else rc(motif))
        else:
            annotation.loc[index, 'motif'] = pd.NA


    annotation = annotation.dropna()
    return annotation
    
def delete_calculate_distance(task):
    index, target, alternative, dir, max_distance = task
    ###print(index)
    min_distance = 1e6
    best_alternative = None
    dir = None

    if dir is None:
        distance = compute_edit_distance(target, alternative)
        if distance < min_distance and distance < max_distance:
            min_distance = distance
            best_alternative = alternative
            dir = '+'
        distance = compute_edit_distance(target, rc(alternative))
        if distance < min_distance and distance < max_distance:
            min_distance = distance
            best_alternative = alternative
            dir = '-'
    elif dir == '+':
        distance = compute_edit_distance(target, alternative)
        if distance < min_distance and distance < max_distance:
            min_distance = distance
            best_alternative = alternative
    elif dir == '-':
        distance = compute_edit_distance(target, rc(alternative))
        if distance < min_distance and distance < max_distance:
            min_distance = distance
            best_alternative = alternative
    else:
        raise ValueError("unrecognized direction!")
        
    return (index, target, best_alternative, dir, min_distance)

def delete(annotation, sources, id2motif, args):
    # get deletion target
    id_to_del = []
    for motif, dir in sources:
        if dir is None:
            id_to_del.extend([f"{motif}+", f"{motif}-"])
        else:
            id_to_del.append(f"{motif}{dir}")

    # get index to revise
    index_list = []
    for source, dir in sources:
        motif = id2motif[source]
        if dir is None:
            mask = (annotation['motif'] == motif)
        else:
            mask = (annotation['motif'] == motif) & (annotation['dir'] == dir)
        index_list.extend(annotation[mask].index.tolist())
    index_list = list(set(index_list))
    target_motifs = annotation.loc[index_list, ]['actual_motif'].tolist()
    
    if len(index_list) == 0:
        return annotation

    # get current motif set
    unique_combinations = annotation[['motif', 'dir']].drop_duplicates()
    motif_list = []
    for _, row in unique_combinations.iterrows():
        tmp = f"{motif2id[row['motif']]}{row['dir']}"
        if tmp not in id_to_del:
            motif_list.append([row['motif'], row['dir']])

    idx_min_dist_dict, idx_result_dict = {}, {}
    for alternative, dir in motif_list:
        max_distance = len(alternative) * threshold
        tasks = [(index_list[i], target_motifs[i], alternative, dir, max_distance) for i in range(len(target_motifs))]
        
        ###print(tasks)

        with Pool(processes=args.thread) as pool:
            results = pool.imap_unordered(delete_calculate_distance, tasks)
            pool.close()
            pool.join()
        ###print("ok!")

        for result in results:
            index, target_motif, best_alternative, dir, min_distance = result
            if index not in idx_min_dist_dict.keys():
                idx_min_dist_dict[index] = min_distance
                idx_result_dict[index] = result
                continue
            if min_distance < idx_min_dist_dict[index]:
                idx_min_dist_dict[index] = min_distance
                idx_result_dict[index] = result

    for index, result in idx_result_dict.items():
        index, target_motif, best_alternative, dir, min_distance = result
        max_distance = len(target_motif) * threshold
        if min_distance < max_distance:
            annotation.loc[index, 'motif'] = best_alternative
            annotation.loc[index, 'dir'] = dir
            annotation.loc[index, 'CIGAR'] = get_cigar(target_motif, best_alternative if dir == '+' else rc(best_alternative))
        else:
            annotation.loc[index, 'motif'] = pd.NA

    annotation = annotation.dropna()
    return annotation

def replace_calculate_distance(task):
    index, target_motif, new_motif, new_dir, max_distance = task

    min_distance = 1e6
    is_replace = None
    if new_dir == '+':
        distance = compute_edit_distance(target_motif, new_motif)
        if distance < min(min_distance, max_distance):
            min_distance = distance
            is_replace = True
    elif new_dir == '-':
        distance = compute_edit_distance(target_motif, rc(new_motif))
        if distance < min(min_distance, max_distance):
            min_distance = distance
            is_replace = True
    else:
        distance = compute_edit_distance(target_motif, new_motif)
        if distance < min(min_distance, max_distance):
            min_distance = distance
            is_replace = True
            new_dir = '+'
        distance = compute_edit_distance(target_motif, rc(new_motif))
        if distance < min(min_distance, max_distance):
            min_distance = distance
            is_replace = True
            new_dir = '-'

    return (index, target_motif, new_motif, is_replace, new_dir)

def replace(annotation, sources, id2motif, args):
    old, new = sources
    old_motif, old_dir = old
    new_motif, new_dir = new

    old_motif = id2motif[old_motif]
    new_motif = id2motif[new_motif]

    # get index to revise
    if old_dir is None:
        mask = (annotation['motif'] == old_motif)
    else:
        mask = (annotation['motif'] == old_motif) & (annotation['dir'] == old_dir)
    index_list = annotation[mask].index.tolist()

    if len(index_list) == 0:
        return annotation

    target_motifs = annotation.loc[index_list, 'actual_motif'].tolist()
    max_distance = len(old_motif) * threshold
    tasks = [(index_list[i], target_motifs[i], new_motif, new_dir, max_distance) for i in range(len(target_motifs))]


    with Pool(processes=args.thread) as pool:
        results = pool.imap_unordered(replace_calculate_distance, tasks)
        pool.close()
        pool.join()

    for index, target_motif, new_motif, is_replace, new_dir in results:
        if is_replace:
            annotation.loc[index, 'motif'] = new_motif
            annotation.loc[index, 'dir'] = new_dir
            annotation.loc[index, 'CIGAR'] = get_cigar(target_motif, new_motif if new_dir == '+' else rc(new_motif))
        else:
            annotation.loc[index, 'motif'] = pd.NA
    
    annotation = annotation.dropna()
    return annotation


# refine.py
def run_refine(args, parser):
    if args.out is None:
        args.out = f'{args.prefix}.revised'
    
    # parse and check action file
    action_df = pd.read_table(args.action, sep = "\t", header = None, names = ["source", "action"])
    actions = parse_and_validate_actions(action_df)

    annotation = pd.read_table(f"{args.prefix}.annotation.tsv")
    motif = pd.read_table(f"{args.prefix}.motif.tsv")
    config = pd.read_json(f"{args.prefix}.setting.json")

    global motif2id
    motif2id = {row['motif']: str(row['id']) for _, row in motif.iterrows()}
    id2motif = {str(row['id']): row['motif'] for _, row in motif.iterrows()}

    global threshold
    threshold = config["Annotation Options"]["annotation_dist_ratio"]

    # revise annotation file
    for action in actions:
        target, type = action
        if type == "MERGE":
            print(f"merging: {target}")
            annotation = merge(annotation, target, id2motif, args)
        elif type == "DELETE":
            print(f"deleting: {target}")
            annotation = delete(annotation, target, id2motif, args)
        elif type == "REPLACE":
            print(f"replacing: {target}")
            annotation = replace(annotation, target, id2motif, args)
        else:
            raise ValueError("invalid action!")

    annotation.to_csv(f"{args.out}.annotation.tsv", sep = '\t', index = False)

    # convert to concise file
    concise_list = []
    merged_cigar, rep_num = '', 0
    for index, row in annotation.iterrows():
        if index != 0:
            if row['seq'] != pre_seq or row['motif'] != pre_motif or row['dir'] != pre_dir:
                concise_list.append([pre_seq, pre_length, start, end, pre_motif, pre_dir, rep_num, '*', '*']) # seq   length  start   end motif   dir rep_num score   CIGAR
                start = row['start']
                rep_num = 0
        else:
            start = row['start']
        pre_seq, pre_length, end, pre_motif, pre_dir = row['seq'], row['length'], row['end'], row['motif'], row['dir']
        rep_num += 1
  
    concise = pd.DataFrame(concise_list, columns = ['seq','length','start','end','motif','dir','rep_num','score','CIGAR'])
    concise.to_csv(f"{args.out}.concise.tsv", sep = '\t', index = False)

    # get revised motif file
    motif['rep_num'] = 0
    name_counts = annotation['motif'].value_counts()

    for name, count in name_counts.items():
        motif.loc[motif['motif'] == name, 'rep_num'] = count

    motif = motif[motif['rep_num'] != 0]
    motif.to_csv(f"{args.out}.motif.tsv", sep = '\t', index = False)

    motifid_list = motif['id'].tolist()
    
    # get revised dist file
    dist = pd.read_table(f"{args.prefix}.dist.tsv")
    selected = [(dist.loc[idx, 'ref'] in motifid_list) and (dist.loc[idx, 'query'] in motifid_list) for idx in range(dist.shape[0])]
    dist = dist.loc[selected,]
    dist.to_csv(f"{args.out}.dist.tsv", sep = '\t', index = False, columns = ['ref','query','dist','is_rc'])
