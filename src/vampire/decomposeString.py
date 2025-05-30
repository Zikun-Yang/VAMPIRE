import sourmash
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import edlib


# calculate layout for graph
def calculate_circular_layout(g):
    offset = 0
    step = 200
    radius = 300
    pos = {}
    num_nodes = len(g.nodes())

    ori_pos = nx.circular_layout(g, scale = 500)

    cycles = list(nx.simple_cycles(g))
    for cycle in cycles:
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint = False)  # calculate angle for each point
        for idx in range(len(cycle)):
            if (cycle[idx] not in pos.keys()):
                pos[cycle[idx]] = np.array([offset + radius*np.cos(angles[idx]), offset + radius*np.sin(angles[idx])])
        offset += step
    for node in pos.keys():
        ori_pos[node] = pos[node]
    return ori_pos

# paint
def paint_dbg(g, pos):
    fig = plt.figure(figsize=(5, 5))
    #edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_nodes(g, pos, node_color = '#A7C957', node_size = 300)
    nx.draw_networkx_edges(
        g, pos, width = 1, 
        connectionstyle="arc3,rad=0.2"  # curved edges
    )
    nx.draw_networkx_labels(g, pos, font_size=5)

    # hide frame
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    #plt.close()
    #plt.savefig(filename,dpi=300)
    return fig

def rotate_strings(s):
    n = len(s)
    return [s[i:] + s[:i] for i in range(n)]

class SparseTable:
    """Sparse Table for Range Minimum Queries (RMQ) on LCP array."""
    def __init__(self, data):
        self.n = len(data)
        self.log = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.log[i] = self.log[i // 2] + 1
        self.k = self.log[self.n]
        self.st = [[0] * self.n for _ in range(self.k + 1)]
        self.st[0] = data
        for i in range(1, self.k + 1):
            j = 0
            while j + (1 << i) <= self.n:
                self.st[i][j] = min(self.st[i - 1][j], self.st[i - 1][j + (1 << (i - 1))])
                j += 1

    def query(self, l, r):
        """Returns the minimum value in the range [l, r]."""
        j = self.log[r - l + 1]
        return min(self.st[j][l], self.st[j][r - (1 << j) + 1])

def rc(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement[base] for base in reversed(seq))

def find_similar_match_plus_chain(seq: str, motifs: list, max_distances: list):

    motif_positions = []
    ###sparse_table = SparseTable(lcp)  # Build Sparse Table for LCP array

    for motif_id, motif in enumerate(motifs):
        ###print(motif_id, motif)
        max_distance = max_distances[motif_id]
        masked_seq = list(seq)  # Convert to list for easy masking
        while True:
            masked_seq_str = ''.join(masked_seq)
            matches = edlib.align(motif, masked_seq_str, mode = "HW", task = "locations", k = max_distance)

            ###print(matches)

            if matches['editDistance'] == -1:
                break

            # Record matches and mask the sequence
            for start, end in matches['locations']:
                ###print(start, end)
                match_start = start
                match_end = end + 1
                motif_positions.append(
                    (match_start, match_end, motif, seq[match_start: match_end], matches['editDistance'])
                )
                # Mask the matched region
                for i in range(match_start, match_end):
                    masked_seq[i] = 'N'  

    motif_match_df = pd.DataFrame(motif_positions, columns=['start', 'end', 'motif', 'seq', 'distance'])
    motif_match_df['dir'] = '+'
    motif_match_df = motif_match_df.sort_values(by=['start']).reset_index(drop=True)

    return motif_match_df
    
def find_similar_match_minor_chain(seq: str, motifs: list, max_distances: list):

    motif_positions = []
    ###sparse_table = SparseTable(lcp)  # Build Sparse Table for LCP array

    for motif_id, motif in enumerate(motifs):
        if len(motif) == 1:
            continue
        ###print(motif_id, motif)
        max_distance = max_distances[motif_id]
        masked_seq = list(seq)  # Convert to list for easy masking
        while True:
            masked_seq_str = ''.join(masked_seq)
            matches = edlib.align(rc(motif), masked_seq_str, mode = "HW", task = "locations", k = max_distance)

            ###print(matches)

            if matches['editDistance'] == -1:
                break

            # Record matches and mask the sequence
            for start, end in matches['locations']:
                ###print(start, end)
                match_start = start
                match_end = end + 1
                motif_positions.append(
                    (match_start, match_end, motif, seq[match_start: match_end], matches['editDistance'])
                )
                # Mask the matched region
                for i in range(match_start, match_end):
                    masked_seq[i] = 'N'  

    motif_match_df = pd.DataFrame(motif_positions, columns=['start', 'end', 'motif', 'seq', 'distance'])
    motif_match_df['dir'] = '-'
    motif_match_df = motif_match_df.sort_values(by=['start']).reset_index(drop=True)

    return motif_match_df



class Decompose:
    def __init__(self, sequence, ksize, args_decomp, args_anno):
        self.sequecne = sequence
        self.ksize = ksize
        # decomposition parameters
        '''
        "Decomposition Options": {
        "ksize": self.args.ksize,
        "motif": self.args.motif,
        "motifnum": self.args.motifnum,
        "force": self.args.force,
        "abud_threshold": self.args.abud_threshold,
        "abud_min": self.args.abud_min,
        "plot": self.args.plot}
        '''
        self.abud_threshold = args_decomp['abud_threshold']
        self.abud_min = args_decomp['abud_min']
        self.is_paint = args_decomp['plot']  # if you want to plot dbg, set True; visualization is unless for complex TR region
        # annotation parameters
        '''
        "Annotation Options": {
        "annotation_dist_ratio": self.args.annotation_dist_ratio,
        "finding_dist_ratio": self.args.finding_dist_ratio}
        '''
        self.dist_ratio = args_anno['annotation_dist_ratio']

        # results
        self.kmer = None
        self.dbg, self.fig = None, None
        self.motifs = None
        self.motifs_list = None
        self.annotation = None

    def count_kmers(self):
        if self.kmer is None:
            mh = sourmash.MinHash(n = 5000, ksize = self.ksize + 1)
            kmers = [kmer for kmer, _ in mh.kmers_and_hashes(self.sequecne, force=True) ]
            kmer_count = Counter(kmers)
            max_count = kmer_count.most_common(1)[0][1]
            min_count = max(self.abud_min, max_count * self.abud_threshold)
            filtered_kmer_count = {k: v for k, v in kmer_count.items() if v >= min_count}
            self.kmer = filtered_kmer_count
        return self.kmer

    def build_dbg(self, prefix, seq_name, pid):
        if self.dbg is None:
            dbg = nx.DiGraph()
            for k in self.kmer:
                dbg.add_edge(k[:-1], k[1:], weight = self.kmer[k])

            if self.is_paint:
                pos = calculate_circular_layout(dbg)
                #paint_dbg(dbg, pos, 'initial_graph.pdf')
            
            # make compact De Bruijn graph
            nodes_to_merge = [n for n in dbg.nodes() if dbg.in_degree(n) == 1 and dbg.out_degree(n) == 1] 

            fig = None
            cot = 1
            for node in nodes_to_merge:
                if node not in dbg.nodes():
                    continue

                # find the start
                added_nodes = [node]
                pre = list(dbg.predecessors(node))[0]
                while dbg.in_degree(pre) == 1 and dbg.out_degree(pre) == 1 and not pre in added_nodes:
                    added_nodes.append(pre)
                    pre = list(dbg.predecessors(pre))[0]
                
                added_nodes = added_nodes[::-1]

                # find the end
                next = list(dbg.successors(node))[0]
                while dbg.in_degree(next) == 1 and dbg.out_degree(next) == 1 and not next in added_nodes:
                    added_nodes.append(next)
                    next = list(dbg.successors(next))[0]

                if len(added_nodes) == 1:    # Simple loop; no merging
                    continue
                
                merged_node = added_nodes[0] + ''.join(node[-1] for node in added_nodes[1:])
                start = added_nodes[0]
                end = added_nodes[-1]
                predecessors = list(dbg.predecessors(start))
                successors = list(dbg.successors(end))

                if dbg.has_edge(end, start): # just a simple loop
                    dbg.add_edge(merged_node, merged_node, weight = dbg[end][start]['weight'])
                else:
                    for predecessor in predecessors:
                        dbg.add_edge(predecessor, merged_node, weight = dbg[predecessor][start]['weight'])
                    for successor in successors:
                        dbg.add_edge(merged_node, successor, weight = dbg[end][successor]['weight'])
                
                # delete original node
                dbg.remove_nodes_from(added_nodes)

                if self.is_paint:
                    pos[merged_node] = pos[start]
                    for n in added_nodes:
                        pos.pop(n, None)
                    fig = paint_dbg(dbg, pos)
                    if fig is not None:
                        plt.savefig(f"{prefix}_temp/{seq_name}_decomp_{pid}.pdf", dpi=300)
                        plt.close(fig)

                cot += 1

            self.dbg = dbg

        return self.dbg

    def find_motif(self):
        if self.motifs is None:
            cycles = nx.simple_cycles(self.dbg)
            motifs = []
            for cycle in cycles:
                # get motif
                motif = ''.join([node[(self.ksize - 1):] for node in cycle])

                '''min_distance = 1e6
                best_motif = motif
                ref_motif = 'UNKNOWN'
                sequences = rotate_strings(motif)  # get all format
                max_distance = int(0.2 * len(motif))
                for seq in sequences:
                    matches = self.ref.find(seq, max_distance)  # find matched motif
                    for dist, ref in matches:
                        if dist < min_distance:
                            best_motif = seq
                            min_distance = dist
                            ref_motif = ref'''

                cot = [self.dbg[cycle[i]][cycle[i+1]]['weight'] for i in range(len(cycle) - 1)]
                cot.append(self.dbg[cycle[-1]][cycle[0]]['weight'])  # add the weight of the last edge

                # print(f'{min_distance}\t{best_motif}\t{ref_motif}\t{min(cot)}')
                motifs.append([motif, 'UNKNOWN', min(cot)])

            motifs_df = pd.DataFrame(motifs, columns=['motif', 'ref_motif', 'value'])
            motifs_df = motifs_df.sort_values(by=['value'], ascending = False).reset_index(drop=True)

            self.motifs = motifs_df
            self.motifs_list = motifs_df['motif'].to_list()
        return self.motifs

    def annotate_with_motif(self):
        if self.annotation is None:
            motifs = self.motifs_list
            motifs_rc = [motif for motif in motifs if motif != rc(motif)]
            max_distances = [ int( len(motif) * self.dist_ratio ) for motif in motifs ]

            motif_match_df = find_similar_match_plus_chain(self.sequecne, motifs, max_distances)
            if len(motifs_rc) > 0:
                motif_rc_match_df = find_similar_match_minor_chain(self.sequecne, motifs_rc, max_distances)
            else:
                motif_rc_match_df = pd.DataFrame()
            result = pd.concat([motif_match_df, motif_rc_match_df], ignore_index = True)    # ['start', 'end', 'motif', 'seq', 'distance','dir']

            # only reserve the row with min distance among rows with the same start and end
            min_indices = result.groupby(['start', 'end'])['distance'].idxmin()
            result = result.loc[min_indices]

            result = result.sort_values(by=['start']).reset_index(drop=True)
            self.annotation = result
        return self.annotation       

    '''def annotate_polish_head(self):
        tmp = []
        max_distance = { motif : int( len(motif) * self.dist_ratio ) for motif in self.motifs_list }
        for motif in self.motifs_list:
            motif_r = motif[::-1]
            seq = self.sequecne[0:len(motif)][::-1]

            for idx in range(len(seq)):
                # print(motif_r, seq[idx:])
                matches = edlib.align(motif_r[0: len(motif) - idx], seq[idx:], mode="SHW", task="locations", k = max_distance[motif])

                if matches['editDistance'] == -1:   # not match
                    continue

                for start, end in matches['locations']:
                    tmp.append(
                        (0, len(seq) - idx, motif, seq[idx:][::-1], matches['editDistance'])
                    )

        tmp = pd.DataFrame(tmp, columns= ['start', 'end', 'motif', 'seq', 'distance'])
        
        tmp = pd.concat([tmp, self.annotation], ignore_index=True)
        tmp = tmp.sort_values(by=['end']).reset_index(drop=True)
        self.annotation = tmp
        return self.annotation'''  

if __name__ == "__main__":
    ##############################
    # config
    ##############################
    k = 5
    seed = 55889615
    
    ##############################
    # count k-mer
    ##############################
    hsat2_hsat3 = TR_multiMotif(['AATGG','CCATT','CTT'], 5000, 0.1, seed)
    print(hsat2_hsat3.sequence[:100])
    example = Decompose(hsat2_hsat3.sequence, k)
    example.count_kmers()
    example.build_dbg()
    example.find_motif()
    example.annotate_with_motif()
    print(example.annotate_with_motif())
    #print(hsat2_hsat3.annotation)
    #example.count_kmers()


