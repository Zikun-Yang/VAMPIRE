import argparse
import pandas as pd
import subprocess
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type = str, required=True, help = "")
parser.add_argument("-a", "--annotation", type = str, required=True, help = "")
parser.add_argument("-o", "--output", type = str, default = None, help = "")
args = parser.parse_args()

def rc(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N':'N'}
    return ''.join(complement[base] for base in reversed(seq))

if args.output is None:
    prefix = args.input.replace('.fasta','').replace('.fa','')
    args.output = f"{prefix}.inverted.fa"


def simple_data(df):
    max_X = max(df['end'].tolist())
    min_X = min(df['start'].tolist())
    result = df.loc[[0]]
    for _, row in df.iterrows():
        last_row = result.iloc[-1]
        last_end = last_row['end']
        last_dir = last_row['dir']
        cur_start = row['start']
        cur_dir = row['dir']
        
        if cur_start - last_end <= (max_X - min_X) * 0.01 and last_dir == cur_dir:
            result.loc[result.shape[0]-1, 'end'] = row['end']
        else:
            result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)

    result = result[result['dir'] == '-']
    result.reset_index(inplace=True, drop = True)

    return result

if __name__ == "__main__":
    
    records = SeqIO.parse(args.input, "fasta")

    annotation = pd.read_table(args.annotation, sep  ='\t')
    annotation.reset_index(inplace = True, drop = True)

    modified_records = list() 
    for record in records:
        print("Seq ID:", record.id)
        #print("Sequence:", record.seq)
        #print("Length:", len(record))
        original_seq = record.seq
        new_seq = list(original_seq)
        tmp = simple_data(annotation[annotation['seq'] == record.id])
        # repolish              #######
        tmp = simple_data(tmp)  #######
        print(tmp)
        
        for _, row in tmp.iterrows():
            start, end = int(row['start']) , int(row['end'])
            segment = original_seq[start:end]
            reverse_complement_segment = segment.reverse_complement()
            new_seq[start:end] = reverse_complement_segment

        new_seq = ''.join(str(base) for base in new_seq)

        modified_records.append(SeqRecord(new_seq, id = record.id))

    SeqIO.write(modified_records, args.output, "fasta")
    subprocess.run(f"sed -i 's/ <unknown description>/_inverted/g' {args.output}", shell = True)