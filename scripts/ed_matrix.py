import argparse
from multiprocessing import Pool
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
import edlib

def rc(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N':'N'}
    return ''.join(complement[base] for base in reversed(seq))


##########################################
# arguments
##########################################
parser = argparse.ArgumentParser(description="calculate edit distance heatmaps between actual motif and reference motif")
parser.add_argument('input', type=str, help='prefix to analyse')
parser.add_argument('output', type=str, help='output prefix')
parser.add_argument('-t','--thread', type=int, default=6, help='thread number')
parser.add_argument('-p','--percentage', type=int, default=75, help='threshold for identifying abnormal values')
parser.add_argument('-s','--show-distance', action='store_true', help='set to show detailed distance on heatmap')
args = parser.parse_args()

for mode in ['raw','normalized']:
    for dir in ['merged', 'separate']:
        print(f'mode={mode}, dir={dir}')
        ##########################################
        # read data
        ##########################################
        annotation = pd.read_table(f"{args.input}.annotation.tsv", sep = '\t', header = 0)
        motif = pd.read_table(f"{args.input}.motif.tsv", sep = '\t', header = 0)

        # get reference motif and their id
        annotated_motif = motif['motif'].to_list()
        motif2id = {row['motif']: f"{row['id']}+" if dir == 'separate' else f"{row['id']}"
                    for _, row in motif.iterrows()}

        motifrc = {}
        if dir == 'separate':
            motif_counts = annotation.groupby(['motif', 'dir']).size().reset_index(name='counts')
            for idx in range(motif_counts.shape[0]):
                if motif_counts.loc[idx, 'dir'] == '-':
                    motif = motif_counts.loc[idx, 'motif']
                    motif_rc = rc(motif)
                    motif2id[motif_rc] = motif2id[motif].replace('+', '-')
                    annotated_motif.append(motif_rc)

        ##########################################
        # calculate in parallels
        ##########################################
        def calculate_dist(task):
            actual_motif, ref_motif = task
            result = [actual_motif, ref_motif]
            for _, motif in enumerate(annotated_motif):
                matches1 = edlib.align(actual_motif, motif, mode = 'HW', task = 'distance')
                matches2 = edlib.align(ref_motif, motif, mode = 'HW', task = 'distance') 
                dist = abs(matches1['editDistance'] - matches2['editDistance']) if mode == 'normalized' else matches1['editDistance']
                result.append(dist)
            return result

        # adjust motif
        if dir == 'merged':
            for i in range(annotation.shape[0]):
                if annotation.loc[i, 'dir'] == '-':
                    annotation.loc[i, 'actual_motif'] = rc(annotation.loc[i, 'actual_motif'])
        else:
            for i in range(annotation.shape[0]):
                if annotation.loc[i, 'dir'] == '-':
                    annotation.loc[i, 'motif'] = rc(annotation.loc[i, 'motif'])

        task = [[row['actual_motif'], row['motif']] for _, row in annotation.iterrows()]
        with Pool(processes=args.thread) as pool:
            results = pool.map(calculate_dist, task)

        results = list(results)
        stats = pd.DataFrame(results, columns = ['actual_motif' ,'ref_motif'] + annotated_motif)

        # get average distances
        average_dist = stats.groupby('ref_motif')[annotated_motif].mean().reset_index()


        ##########################################
        # replace long motif with id
        ##########################################
        def shorten_motif(motif):
            if motif in ['actual_motif', 'ref_motif']:
                return motif
            if motif not in motif2id.keys():
                raise ValueError(f'motif {motif} not in motif2id')
            return str(motif2id[motif])

        # replace names of row and column
        average_dist['ref_motif'] = average_dist['ref_motif'].apply(shorten_motif)
        average_dist.columns = [shorten_motif(col) for col in average_dist.columns]
        # set row and column same
        rownames = average_dist['ref_motif'].to_list()
        average_dist = average_dist.loc[:, ['ref_motif'] + rownames]

        ##########################################
        # save distance matrix
        ##########################################
        average_dist = average_dist.round(2)
        average_dist.to_csv(f"{args.output}_{mode}_{dir}_distance.tsv", sep='\t', index=False)


        average_dist = average_dist.set_index('ref_motif')
        ##########################################
        # plot clustered heatmap
        ##########################################
        # get data matrix
        data_matrix = average_dist.values
        if mode == 'normalized':
            data_matrix = np.abs(data_matrix)

        # cluster by row
        row_linkage = linkage(data_matrix, method='ward', metric='euclidean')
        row_order = leaves_list(row_linkage)

        # set columns index same as row's
        clustered_data = data_matrix[row_order, :][:, row_order]

        # get labels
        row_labels = average_dist.index[row_order].tolist()
        col_labels = row_labels 

        # plot
        ###print(data_matrix.dtype)
        vmax = np.percentile(data_matrix, args.percentage)  # set vmax
        plt.figure(figsize=(12, 8))
        cmap = 'Reds' if mode == 'normalized' else 'Reds_r'
        sns.heatmap(
            clustered_data, 
            vmin=0, 
            vmax=vmax, 
            cmap=cmap, 
            annot=args.show_distance, 
            fmt=".2f",
            xticklabels=col_labels,
            yticklabels=row_labels,
            cbar_kws={'label': 'Mean Edit Distance'},
            square=True,
            annot_kws={'fontsize': 6}
        )
        plt.title(f'Average Edit Distance Heatmap ({mode} distance, dir {dir} )')
        plt.xlabel('Reference Motif')
        plt.ylabel('Actual Motif')
        plt.tight_layout()
        plt.savefig(f'{args.output}_{mode}_{dir}.pdf', dpi = 300)
        plt.close()





