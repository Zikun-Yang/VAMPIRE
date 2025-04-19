import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    description='',
    formatter_class=argparse.RawTextHelpFormatter,
    add_help=True
)

# I/O Options
parser.add_argument('input', type=str, help='.motif.tsv file you want to make')
parser.add_argument('output', type=str, help='output')
args = parser.parse_args()

def parse_motif(file):
    # id, motif, rep_num, label
    motif = pd.read_table(file, sep = '\t', header = 0)
    return motif

motif = parse_motif(args.input)

with open(args.output, 'w') as out:
    for idx in range(motif.shape[0]):
        m = motif.loc[idx, 'motif']
        out.write(f">{motif.loc[idx, 'id']}\n")
        out.write(m + '\n')
