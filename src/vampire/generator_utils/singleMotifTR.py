import numpy as np
import pandas as pd

def mute(motif, mutation_df):
    # config
    let2num = {'A' : 0, 'G' : 1, 'C' : 2, 'T' : 3}
    num2let = {0 : 'A', 1 : 'G', 2 : 'C', 3 : 'T'}

    motif_list = list(motif)
    for index, row in mutation_df.iterrows():
        ###print(row)
        pos = int(row['position'])
        typ = str(row['type'])
        sft = int(row['shift'])
        ###print(pos, typ, sft)
        
        if typ == 'ins':
            motif_list.insert(pos, num2let[sft])
        elif typ == 'del':
            if pos < len(motif_list):
                del motif_list[pos]
        elif typ == 'sub':
            if pos < len(motif_list):
                if not sft:
                    sft = np.random.randint(1, 4)
                motif_list[pos] = num2let[(let2num[motif_list[pos]] + sft) % 4]
    
    mutated_motif = ''.join(motif_list)
    return mutated_motif

class TR_singleMotif:
    def __init__(self, motif, length, mutation_rate, seed):
        self.motif = motif
        self.length = length
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.sequence, self.annotation, self.annotation_woMut = self.generate_seq()
        
    def generate_seq(self):
        np.random.seed(self.seed)
        # generate mutations
        mutation_num = int(self.mutation_rate * self.length)
        if mutation_num > 0:
            mutation_types = ['sub','ins','del']
            random_position = np.random.randint(0, int(self.length * 0.9), size = mutation_num)
            random_type = np.random.choice(mutation_types, size = mutation_num)
            random_shift = np.random.randint(0, 4, size = mutation_num)
            mutation_df = pd.DataFrame({'position':random_position,
                                    'type':random_type,
                                    'shift':random_shift})
            mutation_df = mutation_df.sort_values(by=['position'])
            mutation_df = mutation_df.reset_index(drop=True)

        # generate annotation after mutation
        motif_len = len(self.motif)
        annotation_df = pd.DataFrame(columns=['start','end','motif','rep'])
        cur = 0     # not include cur index
        motif = self.motif
        max_pos = max(mutation_df.loc[:,'position']) if mutation_num > 0 else -1
        while cur <= max_pos:  
            pre_cur = cur
            #print(mutation_df.loc[(mutation_df['position'] >= cur) and (mutation_df['position'] < cur + motif_len),])
            while not mutation_df.loc[(mutation_df['position'] >= cur) & (mutation_df['position'] < cur + motif_len),].shape[0]:
                cur += motif_len
            rep = int((cur - pre_cur) / motif_len)
            if rep:   # rep != 0
                annotation_df.loc[annotation_df.shape[0]] = [pre_cur, cur, self.motif, rep]
            ###print(cur)
            # cur ~ cur + motif_len  has mutation in this region
            mut_tmp = mutation_df.loc[(mutation_df['position'] >= cur) & (mutation_df['position'] < cur + motif_len),]
            mut_tmp.loc[:,'position'] -= cur
            mut_motif = mute(motif, mut_tmp)
            annotation_df.loc[annotation_df.shape[0]] = [cur, cur + len(mut_motif), mut_motif, 1]
            cur += len(mut_motif)
            #print(cur)
        
        # make up the tail of sequence
        if cur != self.length:
            rep = int((self.length - cur) / motif_len)
            annotation_df.loc[annotation_df.shape[0]] = [cur, cur + motif_len * rep, motif, rep]
            cur += motif_len * rep
        if cur != self.length:
            annotation_df.loc[annotation_df.shape[0]] = [cur, self.length, motif[:self.length - cur], 1]

        # generate annotation before mutation
        annotation_woMut_df = pd.DataFrame({'start' : [0], 'end' : [self.length], 'motif' : [self.motif], 'rep': [self.length / len(self.motif)]})

        # generate sequence
        seq = ''
        for i in range(annotation_df.shape[0]):
            seq += annotation_df.loc[i,'motif'] * int(annotation_df.loc[i,'rep'])

        return seq, annotation_df, annotation_woMut_df

    def print_seq(self):
        return self.sequence

    def print_anno(self):
        return self.annotation

    def print_anno_woMut(self):
        return self.annotation_woMut
    
    def save_seq_and_anno(self, prefix):
        width = 60
        with open(prefix + '.fa', 'w') as f:
            f.write('>' + prefix + '\n')
            for i in range(0, len(self.sequence), width):
                f.write(self.sequence[i:i+width] + '\n')
        self.annotation.to_csv(prefix + '.anno.tsv', sep='\t', index=False)
        self.annotation_woMut.to_csv(prefix + '.anno_woMut.tsv', sep='\t', index=False)
