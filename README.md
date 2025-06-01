# VAMPIRE - A tool for annotating the motif variation and complex patterns in tandem repeats.

## <a name="started"></a>Getting Started
```sh
# Install VAMPIRE
pip install vampire

# Annotate STRs with
vampire anno tests/001-anno_STR.fa tests/001-anno_STR

# Generate simulated TR sequences
vampire generator -m GGC -l 1000 -r 0.01 -p tests/002-generator_reference
vampire generator -m GGC GGT -l 1000 -r 0.01 -p tests/002-generator_reference

# Create reference motifset from VAMPIRE annotation files
vampire mkref tests/003-mkref_data tests/003-mkref_reference.fa

# Evaluate the quality of annotation
vampire evaluate tests/001-anno_STR tests/004-evaluate

# Refine the annotation
vampire refine tests/001-anno_STR tests/005-refine_action.tsv -o tests/005-anno_STR.revised

# Plotting sequence logos to visualize motif variation
vampire logo tests/001-anno_STR tests/006-anno_STR_motif
vampire logo --type annotation tests/001-anno_STR tests/006-anno_STR_annotation

# Calculate the identity matrix for TR sequences
vampire identity -w 5 tests/001-anno_STR tests/007-anno_STR
```
See [Cookbook](https://zikun-yang.github.io/VAMPIRE_Cookbook/) for more details.

## <a name="toc"></a>Table of Contents

- [Getting Started](#started)
- [Introduction](#intro)
- [Why VAMPIRE?](#why)
- [Installation](#install)
- [Usage](#usage)
    - [Annotate TR sequences](#anno)
    - [Generate simulated TR sequences](#generator)
    - [Create reference motifset](#mkref)
    - [Evaluate annotation quality](#evaluate)
    - [Refine annotation](#refine)
    - [Plotting sequence logos to visualize motif variation](#logo)
    - [Calculate identity matrix](#identity)
- [Results](#results)
- [Getting Help](#help)
- [Limitations](#limitations)
- [Citing VAMPIRE](#cite)

## <a name="intro"></a>Introduction

VAMPIRE is a unified framework for *de novo* TR motif annotation, structural decomposition, and variation profiling.

## <a name="why"></a>Why VAMPIRE?

#######################################################################
todotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodo
#######################################################################

## <a name="install"></a>Installation
#######################################################################
todotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodotodo
#######################################################################

```sh
# Use singularity (recommended)
singularity pull docker://zikun-yang/vampire:latest #######################################################################

# Install by pip 
pip install vampire # need to install mafft for using logo #######################################################################

# Install by conda
conda install vampire #######################################################################
```

## <a name="usage"></a>Usage

VAMPIRE now contains 7 subcommands: `anno`, `generator`, `mkref`, `evaluate`, `refine`, `logo`, and `identity`.

### <a name="anno"></a>Annotate TR sequences
One basic use of VAMPIRE is to annotate TR sequences in FASTA format. A typical command is as follows:
```sh
# de novo annotate TR sequences
vampire anno -t 8 tests/001-anno_STR.fa tests/001-anno_STR
```
where `-t` sets the number of threads, `tests/001-anno_STR.fa` is the input sequences, and `tests/001-anno_STR` is the output prefix. By default, VAMPIRE use the built-in `base` motif database to refine and label motifs. This database includes pCht/StSat in *Pan* and human alpha-satellite mononers from the paper:
> xxx
> xxx
> xxx
To use a custom motif database, specify it with the `-m` option.

This command will generate five output files:
- `tests/001-anno_STR.settings.json`: annotation parameters used.
- `tests/001-anno_STR.anno.tsv`: detailed annotation, including motif, strand, and actual sequence.
- `tests/001-anno_STR.concise.tsv`: brief annotation results.
- `tests/001-anno_STR.motif.tsv`: motif statistics.
- `tests/001-anno_STR.dist.tsv`: motif distance in plus and minus strands.

Besides, VAMPIRE also supports adding motif database into motif set to annotate TR sequences. 
```sh
# Use motifs from database to annotate (combined with de novo annotation)
vampire anno -f -t 8 [prefix] [output_prefix]

# Only use motifs from database to annotate (without de novo annotation)
vampire anno -f --no-denovo -t 8 [prefix] [output_prefix]
```
For more detailed instructions and examples, refer to [the VAMPIRE Cookbook](https://zikun-yang.github.io/VAMPIRE_Cookbook/).

### <a name="generator"></a>Generate simulated TR sequences
VAMPIRE can generate simulated TR sequences with single or multiple given motif(s), user-defined length and mutation rate. The default random seed is 42. To change the random seed, use the `-s` option.
```sh
# Generate simulated TR sequences
vampire generator -m GGC -l 1000 -r 0.01 -p tests/002-generator_reference
vampire generator -m GGC GGT -l 1000 -r 0.01 -p tests/002-generator_reference
```
This command will output three files:
- `tests/002-generator_reference.fa`: the simulated TR sequences in FASTA format.
- `tests/002-generator_reference.anno.tsv`: the annotation results with mutations.
- `tests/002-generator_reference.fa.anno_woMut.tsv`: the annotation results without mutations.

### <a name="mkref"></a>Create reference motifset

```sh
# Create reference motif set from annotation results
vampire mkref tests/003-mkref_data tests/003-mkref_reference.fa
```
The `mkref` function can generate motif database (in FASTA format) from VAMPIRE annotation results. It can corporate with the `anno` function to annotate TR sequences in a two-step approach: firstly, use `anno` to annotate TR sequences, then use `mkref` to generate motif database from the annotation results. Then, the `anno` function can use this motif database in non-de novo mode to annotate TR sequences.  This two-step approach can generate a motif database on population level and annotate TR sequences with high accuracy.

### <a name="evaluate"></a>Evaluate annotation quality
VAMPIRE evaluates the quality of annotation in a edit distance matrix method. See [the VAMPIRE Cookbook](https://zikun-yang.github.io/VAMPIRE_Cookbook/) for more details.
```sh
# Evaluate the quality of annotation
vampire evaluate tests/001-anno_STR tests/004-evaluate
```
Four figures will be generated:   ##############################################################################
- `tests/004-evaluate.dist.tsv`: the edit distance matrix.
- `tests/004-evaluate.dist.pdf`: the heatmap of the edit distance matrix.
- `tests/004-evaluate.dist.png`: the heatmap of the edit distance matrix.
- `tests/004-evaluate.dist.png`: the heatmap of the edit distance matrix.



### <a name="refine"></a>Refine annotation

```sh
# Refine the annotation
vampire refine tests/001-anno_STR tests/005-refine_action.tsv -o tests/005-anno_STR.revised
```
##############################################################################

### <a name="logo"></a>Plotting sequence logos to visualize motif variation

```sh
# Plotting sequence logos to visualize motif variation
vampire logo tests/001-anno_STR tests/006-anno_STR_motif
vampire logo --type annotation tests/001-anno_STR tests/006-anno_STR_annotation
```
VAMPIRE plots sequence logos in three types: count, probability, and information score. By default, VAMPIRE plot sequence logos using the motif statistics file `*.motif.tsv`. If you want to plot sequence logos using the annotation file `*.anno.tsv` to show the true motif variation, use the `--type annotation` option.

### <a name="identity"></a>Calculate the identity matrix for TR sequences
VAMPIRE uses alignment-based method to calculate the identity matrix for TR sequences.
```sh
# Calculate the identity matrix for TR sequences
vampire identity -t 20 -w 30 tests/001-anno_STR tests/007-anno_STR
```
By default, VAMPIRE do not account for insertion and deletion events when generating the identity matrix. To include such events within a specific length range, use the `--max-indel` and `--min-indel` options to set the maximum and minimum indel lengths to consider.

## <a name="results"></a>Results




## <a name="help"></a>Getting Help



## <a name="limitations"></a>Limitations

## <a name="cite"></a>Citating VAMPIRE

If you use VAMPIRE in your work, please cite:

> To be updated

