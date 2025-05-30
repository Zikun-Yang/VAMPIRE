import numpy as np
import pandas as pd
import subprocess
from Bio.Seq import Seq
from Bio import AlignIO
import subprocess
import matplotlib.pyplot as plt
import logomaker as lm

def rc(seq):
    reverse_complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(reverse_complement[base] for base in reversed(seq))

def invert_annotation(annotation):
    # rc the sequence if dir is -
    for i in range(annotation.shape[0]):
        if annotation.loc[i, 'dir'] == '-':
            annotation.loc[i, 'actual_motif'] = rc(annotation.loc[i, 'actual_motif'])
    return annotation

def remove_N(motif):
    return motif.replace('N', '')

def run_mafft(input_file, output_file=None, options=None):
    cmd = ["mafft"]
    
    if options:
        cmd += options
    
    cmd.append(input_file)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"MAFFT failed:\n{result.stderr}")

    if output_file:
        with open(output_file, 'w') as f:
            f.write(result.stdout)
    else:
        return result.stdout

# run_logo.py
def run_logo(args, parser):
    '''
    module 1: calculate the energy matrix
    '''
    if args.type == 'motif':
        motif = pd.read_table(args.prefix + '.motif.tsv', sep='\t')
        motifs_seq = motif['motif']
        motifs_count = motif['rep_num']

        input_fasta = "temp_input.fasta"
        with open(input_fasta, "w") as fasta_file:
            for i, seq in enumerate(motifs_seq):
                fasta_file.write(f">seq{i}\n{seq}\n")

        # Perform multiple sequence alignment
        output_aln = "temp_output.aln"
        try:
            run_mafft(input_fasta, output_aln)
            print("success to align the motifs:", output_aln)
        except Exception as e:
            print("MAFFT failed:", str(e))
            exit(1)

        # calculate the count matrix manually
        alignment = AlignIO.read(output_aln, "fasta")
        print("alignment:")
        for record in alignment:
            print(record.id, record.seq)

        total_length = len(alignment[0].seq)
        count_matrix = np.zeros((total_length, 6)) # pos, A, C, G, T, -
        char2index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
        for i in range(total_length):
            count_matrix[i][0] = i
            for j in range(len(alignment)):
                seq = alignment[j].seq.upper()
                count_matrix[i, char2index[seq[i]] + 1] += motifs_count[j]

        counts_mat = pd.DataFrame(count_matrix, columns=['pos', 'A', 'C', 'G', 'T', '-'], dtype=int)
        counts_mat.set_index('pos', inplace=True)
        print(counts_mat)
        counts_mat.to_csv(f"{args.output}_count.tsv", sep='\t')

        # remove the temp files
        subprocess.run(f"rm -f {input_fasta} {output_aln}", shell=True)

    elif args.type == 'annotation':
        annotation = pd.read_table(args.prefix + '.annotation.tsv', sep='\t')
        # rc the sequence if dir is -
        annotation = invert_annotation(annotation)
        # filter N character in sequence
        actual_motif = annotation['actual_motif'].apply(remove_N).tolist()
        # count the number of each motif
        actual_motif_count = {motif: actual_motif.count(motif) for motif in list(set(actual_motif))}
        # sort the motifs by count
        actual_motif_count = sorted(actual_motif_count.items(), key=lambda x: x[1], reverse=True)
        motifs_count = [count for _, count in actual_motif_count]
        total_count = sum(motifs_count)

        # align the motifs
        input_fasta = "temp_input.fasta"
        with open(input_fasta, "w") as fasta_file:
            for i, (motif, count) in enumerate(actual_motif_count):
                fasta_file.write(f">seq{i}\n{motif}\n")

        # Perform multiple sequence alignment
        output_aln = "temp_output.aln"
        try:
            run_mafft(input_fasta, output_aln)
            print("success to align the motifs:", output_aln)
        except Exception as e:
            print("MAFFT failed:", str(e))
            exit(1)

        # calculate the count matrix manually
        alignment = AlignIO.read(output_aln, "fasta")
        print("alignment:")
        for i, record in enumerate(alignment):
            print(record.id, record.seq, motifs_count[i])

        # check truncated motifs
        truncated_5prime = {}
        truncated_3prime = {}
        for record in alignment:
            seq = record.seq.upper()
            if seq.startswith('-'):
                for i in range(len(seq)):
                    if seq[i] != '-':
                        break
                if i < len(seq) * 0.1:
                    continue
                truncated_5prime[record.id] = i
            if seq.endswith('-'):
                for i in range(len(seq)-1, -1, -1):
                    if seq[i] != '-':
                        break
                if i < len(seq) * 0.1:
                    continue
                truncated_3prime[record.id] = i

        ###print(truncated_5prime)
        ###print(truncated_3prime)

        # count
        print(f"total_count: {total_count}")
        total_length = len(alignment[0].seq)
        count_matrix = np.zeros((total_length, 6)) # pos, A, C, G, T, -
        char2index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
        for i in range(total_length):
            count_matrix[i][0] = i
            for j in range(len(alignment)):
                seq = alignment[j].seq.upper()
                name = alignment[j].id
                if motifs_count[j] < max(3, total_count * 0.1) and (name in truncated_5prime.keys() or name in truncated_3prime.keys()):
                    if name in truncated_5prime.keys() and i < truncated_5prime[name]:
                        continue
                    if name in truncated_3prime.keys() and i > truncated_3prime[name]:
                        continue
                    ###print(f"motif {name} is truncated at 5' end, count: {motifs_count[j]}")
                count_matrix[i, char2index[seq[i]] + 1] += motifs_count[j]

        counts_mat = pd.DataFrame(count_matrix, columns=['pos', 'A', 'C', 'G', 'T', '-'], dtype=int)
        counts_mat.set_index('pos', inplace=True)
        print(counts_mat)
        counts_mat.to_csv(f"{args.output}_count.tsv", sep='\t')

        subprocess.run(f"rm -f {input_fasta} {output_aln}", shell=True)
    else:
        raise ValueError("Type error! please use motif or annotation")


    '''
    module 2: plot seq logo
    '''

    custom_colors = {
        'A': '#2a9d8f',  # green
        'C': '#00b4d8',  # blue
        'G': '#f4d35e',  # yellow
        'T': '#e63946',   # red
        '-': '#000000'    # black
    }

    lm.Logo(counts_mat, 
            color_scheme=custom_colors,
            figsize=(total_length,3))
    plt.savefig(f"{args.output}_count.{args.format}", format=args.format)

    prob_mat = lm.transform_matrix(counts_mat, from_type='counts', to_type='probability')
    lm.Logo(prob_mat, 
            color_scheme=custom_colors,
            figsize=(total_length,3))
    plt.savefig(f"{args.output}_prob.{args.format}", format=args.format)

    info_mat = lm.transform_matrix(counts_mat, 
                                        from_type='counts', 
                                        to_type='information')
    lm.Logo(info_mat, 
            color_scheme=custom_colors,
            figsize=(total_length,3))
    plt.savefig(f"{args.output}_info.{args.format}", format=args.format)



