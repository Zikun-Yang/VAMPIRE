from typing import Any
import numpy as np
import polars as pl
import logging

logger = logging.getLogger(__name__)

def run_generator(cfg: dict[str, Any]) -> None:
    '''
    Generate simulated tandem repeat sequences

    Parameters
    ----------
        motifs: list[str]
            List of motifs to generate
        length: int
            Length of the tandem repeat
        mutation_rate: float
            Mutation rate, 0 - 1
        seed: int
            Random seed
        prefix: str
            Prefix for the output files

    Returns
    -------
        None
    
    Generates the following files:
        - <prefix>.fa # sequence file
        - <prefix>.anno.tsv # annotation file with actual sequence
        - <prefix>.anno_woMut.tsv # annotation file without mutations
    '''

    motif_list_len = len(cfg['motifs'])
    motifs = cfg['motifs']

    # check invalid characters
    for motif in motifs:
        if not all(c in 'ACGT' for c in motif):
            raise ValueError("ERROR: Invalid characters in motif!")

    if motif_list_len == 1:
        tr = TR_singleMotif(motifs[0], cfg['length'], cfg['mutation_rate'], cfg['seed'])
    else:
        tr = TR_multiMotif(motifs, cfg['length'], cfg['mutation_rate'], cfg['seed'])

    tr.save_seq_and_anno(cfg['prefix'])

def mutate(motif: str, mutation_df: pl.DataFrame) -> str:
    """
    Apply mutations INSIDE a single motif (local coordinate system)
    """
    let2num = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
    num2let = {0: 'A', 1: 'G', 2: 'C', 3: 'T'}

    motif_list = list(motif)

    offset = 0  # compensate the position change caused by indel

    for row in mutation_df.iter_rows(named=True):
        pos: int = int(row["position"]) + offset
        typ: str = str(row["type"])
        sft: int = int(row["shift"])

        if typ == 'ins':
            if 0 <= pos <= len(motif_list):
                motif_list.insert(pos, num2let[sft])
                offset += 1

        elif typ == 'del':
            if 0 <= pos < len(motif_list) and len(motif_list) > 1:
                del motif_list[pos]
                offset -= 1

        elif typ == 'sub':
            if 0 <= pos < len(motif_list):
                if sft == 0:
                    sft = np.random.randint(1, 4)
                motif_list[pos] = num2let[(let2num[motif_list[pos]] + sft) % 4]

    return "".join(motif_list)


class TR_singleMotif:
    def __init__(self, motif, length, mutation_rate, seed):
        self.motif = motif
        self.length = length
        self.mutation_rate = mutation_rate
        self.seed = seed
        logger.debug(f"Initialized with motif: {motif}, length: {length}, mutation_rate: {mutation_rate}, seed: {seed}")
        self.sequence, self.annotation, self.annotation_woMut = self.generate_seq()
        self.validate_annotation()

    def generate_seq(self):
        np.random.seed(self.seed)
        motif = self.motif
        motif_len = len(motif)
        mutation_num = int(self.mutation_rate * self.length)

        # generate mutation table
        if mutation_num > 0:
            mutation_types = np.array(['sub', 'ins', 'del'])
            positions = np.random.randint(0, self.length, size=mutation_num)
            types = np.random.choice(mutation_types, size=mutation_num)
            shifts = np.random.randint(0, 4, size=mutation_num)

            order = np.argsort(positions)
            positions = positions[order]
            types = types[order]
            shifts = shifts[order]

            mutation_df = pl.DataFrame({
                'position': positions.tolist(),
                'type': types.tolist(),
                'shift': shifts.tolist()
            })
        else:
            mutation_df = pl.DataFrame({
                'position': pl.Series([], dtype=pl.Int64),
                'type': pl.Series([], dtype=pl.Utf8),
                'shift': pl.Series([], dtype=pl.Int64)
            })

        annotation = []
        seq_parts = []
        cur = 0
        offset = 0  # for insert/del adjustment in mutation
        copy_count = 1
        while cur < self.length:
            # current motif mutation
            motif_start = cur
            motif_end = min(cur + motif_len, self.length)
            mut_mask = (mutation_df['position'] >= motif_start) & (mutation_df['position'] < motif_end)
            mut_tmp = mutation_df.filter(mut_mask)

            # adjust mutation coordinates to motif local
            mut_tmp = mut_tmp.with_columns([
                (pl.col('position') - motif_start).alias('position')
            ])

            motif_seq = mutate(motif[:motif_end - motif_start], mut_tmp)

            remaining = self.length - cur
            if len(motif_seq) > remaining:
                motif_seq = motif_seq[:remaining]

            seq_parts.append(motif_seq)
            annotation.append((cur, cur + len(motif_seq), motif_seq, 1))
            logger.debug(f"copy {copy_count}, coordinates: {cur}-{cur + len(motif_seq)}, motif: {motif_seq}")
            cur += len(motif_seq)
            copy_count += 1
        # concatenate complete sequence
        seq = ''.join(seq_parts)
        annotation_df = pl.DataFrame(
            annotation,
            schema=["start", "end", "motif", "rep"],
            orient='row'
        )
        annotation_woMut_df = pl.DataFrame({
            'start': pl.Series([0], dtype=pl.Int64),
            'end': pl.Series([self.length], dtype=pl.Int64),
            'motif': pl.Series([motif], dtype=pl.Utf8),
            'rep': pl.Series([self.length / motif_len], dtype=pl.Float16)
        })

        return seq, annotation_df, annotation_woMut_df

    def validate_annotation(self):
        for row in self.annotation.rows():
            s, e, motif, _ = row
            seq_slice = self.sequence[int(s):int(e)]
            assert seq_slice == motif, f"Annotation mismatch {s}-{e}: {seq_slice} != {motif}"
        assert len(self.sequence) == self.length, f"Sequence length {len(self.sequence)} != {self.length}"

    def save_seq_and_anno(self, prefix):
        width = 60
        with open(prefix+'.fa','w') as f:
            f.write('>'+prefix+'\n')
            for i in range(0,len(self.sequence),width):
                f.write(self.sequence[i:i+width]+'\n')
        self.annotation.write_csv(prefix+'.anno.tsv', separator='\t', include_header=True)
        self.annotation_woMut.write_csv(prefix+'.anno_woMut.tsv', separator='\t', include_header=True)


class TR_multiMotif:
    def __init__(self, motif_list, length, mutation_rate, seed):
        self.motif_list = motif_list
        self.length = length
        self.mutation_rate = mutation_rate
        self.seed = seed
        
        self.sequence, self.annotation, self.annotation_woMut = self.generate_seq()
        self.validate_annotation()

    def generate_seq(self):
        logger = logging.getLogger(__name__)
        np.random.seed(self.seed)

        seq = ''
        annotation_df = pl.DataFrame({
            'start': pl.Series([], dtype=pl.Int64),
            'end': pl.Series([], dtype=pl.Int64),
            'motif': pl.Series([], dtype=pl.Utf8),
            'rep': pl.Series([], dtype=pl.Int64)
        })
        annotation_woMut_df = pl.DataFrame({
            'start': pl.Series([], dtype=pl.Int64),
            'end': pl.Series([], dtype=pl.Int64),
            'motif': pl.Series([], dtype=pl.Utf8),
            'rep': pl.Series([], dtype=pl.Float16)
        })

        l = self.length
        motif_repeats = np.random.randint(1,4, size=len(self.motif_list))
        repeated_motifs = []
        for motif, repeat in zip(self.motif_list, motif_repeats):
            repeated_motifs.extend([motif]*repeat)
        np.random.shuffle(repeated_motifs)

        num_motifs = len(repeated_motifs)
        random_numbers = np.random.dirichlet(np.ones(num_motifs), size=1)[0]
        sublen = (random_numbers*l).astype(int)
        sublen[-1] = l - sum(sublen[:-1])

        cur = 0
        for i in range(len(repeated_motifs)):
            logger.debug(f"Generating subTR {i} with consensus motif: {repeated_motifs[i]}, length: {min(sublen[i],l)}")
            subTR = TR_singleMotif(repeated_motifs[i], min(sublen[i],l), self.mutation_rate, self.seed)
            seq += subTR.sequence
            sub_anno = subTR.annotation.with_columns([
                (pl.col("start")+cur).alias("start"),
                (pl.col("end")+cur).alias("end")
            ])
            sub_anno_woMut = subTR.annotation_woMut.with_columns([
                (pl.col("start")+cur).alias("start"),
                (pl.col("end")+cur).alias("end")
            ])
            annotation_df = pl.concat([annotation_df, sub_anno])
            annotation_woMut_df = pl.concat([annotation_woMut_df, sub_anno_woMut])
            cur += sublen[i]
            l -= sublen[i]

        return seq, annotation_df, annotation_woMut_df

    def validate_annotation(self):
        for row in self.annotation.rows():
            s, e, motif, _ = row
            seq_slice = self.sequence[int(s):int(e)]
            assert seq_slice == motif, f"Annotation mismatch {s}-{e}: {seq_slice} != {motif}"
        assert len(self.sequence) == self.length, f"Sequence length {len(self.sequence)} != {self.length}"

    def save_seq_and_anno(self, prefix):
        width = 60
        with open(prefix+'.fa','w') as f:
            f.write(f">{prefix}\n")
            for i in range(0,len(self.sequence),width):
                f.write(f"{self.sequence[i:i+width]}\n")
        self.annotation.write_csv(prefix+'.anno.tsv', separator='\t', include_header=True)
        self.annotation_woMut.write_csv(prefix+'.anno_woMut.tsv', separator='\t', include_header=True)