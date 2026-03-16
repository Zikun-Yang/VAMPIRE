import numpy as np
import pandas as pd


def mute(motif, mutation_df):
    # config
    let2num = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
    num2let = {0: 'A', 1: 'G', 2: 'C', 3: 'T'}

    motif_list = list(motif)

    for pos, typ, sft in zip(
        mutation_df['position'].values,
        mutation_df['type'].values,
        mutation_df['shift'].values
    ):

        pos = int(pos)
        sft = int(sft)

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

    return ''.join(motif_list)


class TR_singleMotif:

    def __init__(self, motif, length, mutation_rate, seed):
        self.motif = motif
        self.length = length
        self.mutation_rate = mutation_rate
        self.seed = seed

        self.sequence, self.annotation, self.annotation_woMut = self.generate_seq()

    def generate_seq(self):

        np.random.seed(self.seed)

        motif = self.motif
        motif_len = len(motif)
        length = self.length

        mutation_num = int(self.mutation_rate * length)

        if mutation_num > 0:

            mutation_types = np.array(['sub', 'ins', 'del'])

            positions = np.random.randint(0, int(length * 0.9), size=mutation_num)
            types = np.random.choice(mutation_types, size=mutation_num)
            shifts = np.random.randint(0, 4, size=mutation_num)

            order = np.argsort(positions)

            positions = positions[order]
            types = types[order]
            shifts = shifts[order]

        else:

            positions = np.array([], dtype=int)
            types = np.array([])
            shifts = np.array([])

        annotation = []
        seq_parts = []

        cur = 0
        m_idx = 0

        max_pos = positions.max() if mutation_num > 0 else -1

        while cur <= max_pos:

            pre_cur = cur

            # skip past mutation
            while m_idx < mutation_num and positions[m_idx] < cur:
                m_idx += 1

            # skip motif block without mutation
            while (
                m_idx < mutation_num
                and positions[m_idx] >= cur + motif_len
            ):
                cur += motif_len

                if cur > max_pos:
                    break

            if m_idx >= mutation_num:
                break

            rep = (cur - pre_cur) // motif_len

            if rep > 0:
                annotation.append((pre_cur, cur, motif, rep))
                seq_parts.append(motif * rep)

            if cur > max_pos:
                break

            # current motif with mutation
            mut_start = m_idx

            while (
                m_idx < mutation_num
                and positions[m_idx] < cur + motif_len
            ):
                m_idx += 1

            mut_end = m_idx

            mut_tmp = pd.DataFrame({
                "position": positions[mut_start:mut_end] - cur,
                "type": types[mut_start:mut_end],
                "shift": shifts[mut_start:mut_end]
            })

            mut_motif = mute(motif, mut_tmp)

            annotation.append((cur, cur + len(mut_motif), mut_motif, 1))
            seq_parts.append(mut_motif)

            cur += len(mut_motif)

        # tail motif
        if cur < length:

            rep = (length - cur) // motif_len

            if rep > 0:
                annotation.append((cur, cur + motif_len * rep, motif, rep))
                seq_parts.append(motif * rep)

                cur += motif_len * rep

        if cur < length:

            tail = motif[:length - cur]

            annotation.append((cur, length, tail, 1))
            seq_parts.append(tail)

        annotation_df = pd.DataFrame(
            annotation,
            columns=['start', 'end', 'motif', 'rep']
        )

        annotation_woMut_df = pd.DataFrame({
            'start': [0],
            'end': [length],
            'motif': [motif],
            'rep': [length / motif_len]
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