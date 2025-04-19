import numpy as np
import pandas as pd

# config
let2num = {'A' : 0, 'G' : 1, 'C' : 2, 'T' : 3}
num2let = {0 : 'A', 1 : 'G', 2 : 'C', 3 : 'T'}

def get_seq(annotation_df):
    seq = ''
    for i in range(annotation_df.shape[0]):
        seq += annotation_df.loc[i,'motif'] * annotation_df.loc[i,'rep']

    return seq


def mute(motif, mutation_df):
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

######################################
# TR_singleMotif
######################################
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

        ###print(annotation_df)

        # generate annotation before mutation
        annotation_woMut_df = pd.DataFrame({'start' : [0], 'end' : [self.length], 'motif' : [self.motif], 'rep': [self.length / len(self.motif)]})
        ###print(annotation_woMut_df)

        # generate sequence
        seq = ''
        for i in range(annotation_df.shape[0]):
            ###print(annotation_df.loc[i,'motif'] , annotation_df.loc[i,'rep'])
            ###print(annotation_df.loc[i,'motif'] * annotation_df.loc[i,'rep'])
            ###print(i)
            ###print('###########3',seq)
            seq += annotation_df.loc[i,'motif'] * int(annotation_df.loc[i,'rep'])
        ###print(seq)

        return seq, annotation_df, annotation_woMut_df

    def print_seq(self):
        return self.sequence

    def print_anno(self):
        return self.annotation

    def print_anno_woMut(self):
        return self.annotation_woMut
    
    def write_seq():
        pass

######################################
# TR_multiMotif
######################################
class TR_multiMotif:
    def __init__(self, motif_list, length, mutation_rate, seed):
        self.motif_list = motif_list
        self.length = length
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.sequence, self.annotation, self.annotation_woMut = self.generate_seq()

    def generate_seq(self):
        np.random.seed(self.seed)
        seq = ''
        annotation_df = pd.DataFrame(columns=['start','end','motif','rep'])
        annotation_woMut_df = pd.DataFrame(columns=['start','end','motif','rep'])

        # determine the organization of complex TR region
        l = self.length
        # Randomly decide how many times each motif repeats (1 to 3 times)
        motif_repeats = np.random.randint(1, 4, size = len(self.motif_list))
        repeated_motifs = []
        for motif, repeat in zip(self.motif_list, motif_repeats):
            repeated_motifs.extend([motif] * repeat)
        # Shuffle the order of the repeated motifs
        np.random.shuffle(repeated_motifs)

        num_motifs = len(repeated_motifs)
        random_numbers = np.random.dirichlet(np.ones(num_motifs), size=1)[0]
        sublen = (random_numbers * l).astype(int)
        # Adjust the last element to ensure the sum is exactly l
        sublen[-1] = l - sum(sublen[:-1])

        # use TR_singleMotif to generate TR with single motif
        cur = 0
        for i in range(len(repeated_motifs)):
            ### print(repeated_motifs[i], min(sublen[i],l), self.mutation_rate, self.seed)
            subTR = TR_singleMotif(repeated_motifs[i], min(sublen[i],l), self.mutation_rate, self.seed)
            seq += subTR.sequence
            sub_anno = subTR.annotation
            ### print(subTR.annotation)
            sub_anno_woMut = subTR.annotation_woMut
            # transform the coordinates
            sub_anno.loc[:,'start'] += cur
            sub_anno.loc[:,'end'] += cur
            sub_anno_woMut.loc[:,'start'] += cur
            sub_anno_woMut.loc[:,'end'] += cur
            # merge 
            if not annotation_df.shape[0]:
                annotation_df = sub_anno
            else:
                annotation_df = pd.concat([annotation_df, sub_anno], ignore_index=True)
            if not annotation_woMut_df.shape[0]:
                annotation_woMut_df = sub_anno_woMut
            else:
                annotation_woMut_df = pd.concat([annotation_woMut_df, sub_anno_woMut], ignore_index=True)
            cur += sublen[i]
            l -= sublen[i]
        
        return seq, annotation_df, annotation_woMut_df

    def print_seq(self):
        return self.sequence

    def print_anno(self):
        return self.annotation

    def print_anno_woMut(self):
        return self.annotation_woMut


if __name__ == "__main__":
    seed = 55889615
    #hsat3 = TR_singleMotif('AATGG', 1000, 0, seed)
    #print(hsat3.sequence)
    #print(hsat3.annotation)
    #print(len(hsat3.sequence))
    #print(hsat3.annotation_woMut)

    '''
    matcher = SequenceMatcher(None, seq, seq_2)
    dif = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        print(tag)
        if tag:
            dif.extend(range(i1,i2))
    #print(dif)
    '''

    #hsat3 = TR_singleMotif('CCATT', 7863, 0.1, seed)

    hsat2_hsat3 = TR_multiMotif(['AATGG','CCATT','TTTA','GGC'], 5000, 0, seed)
    #print(hsat2_hsat3.sequence)
    #print(hsat2_hsat3.annotation)
    #print(hsat2_hsat3.annotation_woMut)
    #print(hsat2_hsat3.annotation)
    #print(len(hsat2_hsat3.sequence))
    #print(hsat2_hsat3.annotation_woMut)



