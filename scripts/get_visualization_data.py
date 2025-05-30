import re
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--prefix', type = str, help = 'TR annotation file')
parser.add_argument('-r', '--repeat', type = str, help = '.out file from TRF')
parser.add_argument('-o', '--output', type = str, help = 'output name')
args = parser.parse_args()

# read annotation data
annotation = pd.read_table(f"{args.prefix}.annotation.tsv", sep = '\t')
motif = pd.read_table(f"{args.prefix}.motif.tsv", sep = '\t')

motif2id = dict()
for index, row in motif.iterrows():
    motif2id[row['motif']] = row['id']

tmp = list()
for index, row in annotation.iterrows():
    tmp.append([row['seq'], row['start'], row['end'], row['dir'], 'TR', motif2id[row['motif']]])

tr_df = pd.DataFrame(tmp, columns = ['name', 'start', 'end', 'dir', 'type', 'annotation'])

# read and parse repeatmasker result
def parse_repeatmasker_output(file_path):
    df = pd.read_csv(file_path, sep = r'\s+', skiprows = 3, header = None, names = ['score', 'perc_div', 'perc_del', 'perc_ins', 
                                                                       'query', 'query_position_start', 'query_position_end', 'matching_repeat_left', 
                                                                       'strand', 'repeat_class', 'repeat_family_and_name',
                                                                        'repeat_position_start', 'repeat_position_end', 'repeat_left', 'repeat_id','other'])
    df[['repeat_family', 'repeat_name']] = df['repeat_family_and_name'].str.split('/', expand=True)
    df.loc[df['strand'] == 'C', 'strand'] = '-'
    return df


df = parse_repeatmasker_output(args.repeat)
tmp = list()
for index, row in df.iterrows():
    tmp.append([row['query'], 
                row['query_position_start'], 
                row['query_position_end'], 
                row['strand'], 
                'TE', 
                row['repeat_family']])
te_df = pd.DataFrame(tmp, columns = ['name', 'start', 'end', 'dir', 'type', 'annotation'])

# merge data
merged_df = pd.concat([tr_df, te_df], ignore_index = True)


# output
merged_df.to_csv(args.output, sep = '\t', index = False)
