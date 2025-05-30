# generator.py
def run_generator(args, parser):
    '''
    version: 0.1.0
    description: generate tandem repeat sequences
    input: motifs, length, mutation rate, seed
    output: prefix (tandem repeat sequences in fasta format, annotation in tsv format)
    '''

    from vampire.generator_utils import TR_singleMotif, TR_multiMotif

    motif_list_len = len(args.motifs)
    motifs = args.motifs

    # check invalid characters
    for motif in motifs:
        if not all(c in 'ACGT' for c in motif):
            raise ValueError("ERROR: Invalid characters in motif!")

    if motif_list_len == 1:
        tr = TR_singleMotif(motifs[0], args.length, args.mutation_rate, args.seed)
    else:
        tr = TR_multiMotif(motifs, args.length, args.mutation_rate, args.seed)

    tr.save_seq_and_anno(args.prefix)
