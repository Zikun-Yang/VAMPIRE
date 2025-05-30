# mkref.py
def run_mkref(args, parser):
    '''
    version: 0.1.0
    description: make reference database (.fa file) from annotation result (.motif.tsv file)
    input: annotation result prefix
    output: reference database (fasta format)
    '''

    import pandas as pd

    filename = args.prefix + '.motif.tsv'
    #columns: id, motif, rep_num, label
    motif = pd.read_table(filename, sep = '\t', header = 0)

    with open(args.output, 'w') as out:
        for idx in range(motif.shape[0]):
            m = motif.loc[idx, 'motif']
            out.write(f">{motif.loc[idx, 'id']}\n{m}\n")
