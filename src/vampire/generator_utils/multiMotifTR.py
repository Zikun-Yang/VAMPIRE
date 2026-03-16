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

        mutation_num = int(self.mutation_rate * self.length)

        if mutation_num > 0:
            mutation_types = np.array(['sub', 'ins', 'del'])

            positions = np.random.randint(0, int(self.length * 0.9), size=mutation_num)
            types = np.random.choice(mutation_types, size=mutation_num)
            shifts = np.random.randint(0, 4, size=mutation_num)

            order = np.argsort(positions)
            positions = positions[order]
            types = types[order]
            shifts = shifts[order]

            mutation_df = pd.DataFrame({
                'position': positions,
                'type': types,
                'shift': shifts
            })
        else:
            mutation_df = pd.DataFrame(columns=['position', 'type', 'shift'])
            positions = np.array([])

        motif = self.motif
        motif_len = len(motif)

        annotation = []
        seq_parts = []

        cur = 0
        max_pos = positions.max() if mutation_num > 0 else -1
        m_idx = 0

        while cur <= max_pos:

            pre_cur = cur

            while True:
                if m_idx >= mutation_num:
                    cur = max_pos + 1
                    break

                pos = positions[m_idx]

                if pos < cur:
                    m_idx += 1
                    continue

                if pos >= cur + motif_len:
                    cur += motif_len
                    continue

                break

            rep = (cur - pre_cur) // motif_len

            if rep:
                annotation.append((pre_cur, cur, motif, rep))
                seq_parts.append(motif * rep)

            if cur > max_pos:
                break

            mut_mask = (positions >= cur) & (positions < cur + motif_len)
            mut_tmp = mutation_df.loc[mut_mask].copy()
            mut_tmp.loc[:, 'position'] -= cur

            mut_motif = mute(motif, mut_tmp)

            annotation.append((cur, cur + len(mut_motif), mut_motif, 1))
            seq_parts.append(mut_motif)

            cur += len(mut_motif)

        if cur != self.length:
            rep = (self.length - cur) // motif_len
            if rep:
                annotation.append((cur, cur + motif_len * rep, motif, rep))
                seq_parts.append(motif * rep)
                cur += motif_len * rep

        if cur != self.length:
            tail = motif[:self.length - cur]
            annotation.append((cur, self.length, tail, 1))
            seq_parts.append(tail)

        annotation_df = pd.DataFrame(
            annotation,
            columns=['start', 'end', 'motif', 'rep']
        )

        annotation_woMut_df = pd.DataFrame({
            'start': [0],
            'end': [self.length],
            'motif': [self.motif],
            'rep': [self.length / motif_len]
        })

        seq = ''.join(seq_parts)

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
    
    def save_seq_and_anno(self, prefix):
        width = 60
        with open(prefix + '.fa', 'w') as f:
            f.write('>' + prefix + '\n')
            for i in range(0, len(self.sequence), width):
                f.write(self.sequence[i:i+width] + '\n')
        self.annotation.to_csv(prefix + '.anno.tsv', sep='\t', index=False)
        self.annotation_woMut.to_csv(prefix + '.anno_woMut.tsv', sep='\t', index=False)