# VAMPIRE
[![PyPI version](https://badge.fury.io/py/vampire-tr.svg)](https://pypi.org/project/vampire-tr/)  [![Docker Image Version](https://img.shields.io/badge/docker-v0.3.0-blue)](https://hub.docker.com/r/zikunyang/vampire-tr/tags) [![License](https://img.shields.io/github/license/Zikun-Yang/VAMPIRE)](https://github.com/Zikun-Yang/VAMPIRE/blob/main/LICENSE)  [![Last Commit](https://img.shields.io/github/last-commit/Zikun-Yang/VAMPIRE)](https://github.com/Zikun-Yang/VAMPIRE/commits/main)

## <a name="started"></a>Getting Started
```sh
# Use singularity / apptainer to pull the prebuilt image from Docker Hub (recommended)
singularity pull docker://zikunyang/vampire-tr:latest
singularity exec vampire-tr_latest.sif vampire --help
# OR build from the definition file in the repository (recommended)
git clone git@github.com:Zikun-Yang/VAMPIRE.git
cd VAMPIRE
singularity build vampire-tr_latest.sif vampire.def
singularity exec vampire-tr_latest.sif vampire --help
# Use docker
docker pull zikunyang/vampire-tr:latest
docker run -it --name vampire-tr zikunyang/vampire-tr:latest
docker exec vampire-tr vampire --help
# Install by pip 
conda create -n vampire python=3.10 -y
conda activate vampire
conda install -c bioconda mafft # you need to install mafft for using logo function
pip install vampire-tr 

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
- [Getting Help](#help)
- [Citing VAMPIRE](#cite)

## <a name="intro"></a>Introduction

VAMPIRE is a unified framework for *de novo* [tandem repeat (TR)](https://en.wikipedia.org/wiki/Tandem_repeat) motif annotation, structural decomposition, and variation profiling.

## <a name="why"></a>Why VAMPIRE?

- **Beyond Copy Number:** VAMPIRE uncovers not only copy number but also internal variation.
- **Flexible and Comprehensive:** Its customizable parameters support the annotation of a wide range of TRs, from short tandem repeats (STRs) and variable number tandem repeats (VNTRs) to megabase-scale satellite arrays.
- **Research-Ready Output:** VAMPIRE generates rich, structured results in standard `.tsv` format—ideal for downstream analysis and integration into custom pipelines.

## <a name="install"></a>Installation

```sh
# Use singularity / apptainer to pull the prebuilt image from Docker Hub (recommended)
singularity pull docker://zikunyang/vampire-tr:latest
singularity exec vampire-tr_latest.sif vampire --help
# OR build from the definition file in the repository (recommended)
git clone git@github.com:Zikun-Yang/VAMPIRE.git
cd VAMPIRE
singularity build vampire-tr_latest.sif vampire.def
singularity exec vampire-tr_latest.sif vampire --help

# Use docker
docker pull zikunyang/vampire-tr:latest
docker run -it --name vampire-tr zikunyang/vampire-tr:latest
docker exec vampire-tr vampire --help

# Install by pip 
conda create -n vampire python=3.10 -y
conda activate vampire
conda install -c bioconda mafft # you need to install mafft for using logo function
pip install vampire-tr 
```

 Add shortcut if you use the container (Optional)
 ```bash
# Open your ~/.bashrc (or ~/.zshrc) and add the following lines:

# For Singularity users:
alias vampire="singularity exec /path/to/your/container/vampire-tr_latest.sif vampire"

# For Docker users (assuming a running container named 'vampire-tr'):
alias vampire="docker exec vampire-tr vampire"

# Save the file, then apply the changes:
source ~/.bashrc
```



    
## <a name="usage"></a>Usage

VAMPIRE now contains 7 subcommands: `anno`, `generator`, `mkref`, `evaluate`, `refine`, `logo`, and `identity`.

### <a name="anno"></a>anno - Annotate TR sequences

One of the primary uses of VAMPIRE is to annotate tandem repeat (TR) sequences from input files in FASTA format. A typical command is as follows:
```sh
# de novo annotate TR sequences
vampire anno -t 8 tests/001-anno_STR.fa tests/001-anno_STR
```
where `-t` sets the number of threads, `tests/001-anno_STR.fa` is the input sequences, and `tests/001-anno_STR` is the output prefix. By default, VAMPIRE use the built-in `base` motif database to refine and label motifs. This database includes pCht/StSat in *Pan* and human alpha-satellite mononers from the paper:
> Altemose N, Logsdon G A, Bzikadze A V, et al. 
> Complete genomic and epigenetic maps of human centromeres[J]. 
> Science, 2022, 376(6588): eabl4178.

To use a custom motif database, specify it with the `-m` option.

This command will generate five output files:
- `tests/001-anno_STR.settings.json`: annotation parameters used.
- `tests/001-anno_STR.anno.tsv`: detailed annotation, including motif, strand, and actual sequence.
- `tests/001-anno_STR.concise.tsv`: brief annotation results.
- `tests/001-anno_STR.motif.tsv`: motif statistics.
- `tests/001-anno_STR.dist.tsv`: motif distance in plus and minus strands.

Besides, VAMPIRE also supports adding motif database into motif set to annotate TR sequences in both de novo and non-de novo modes. 
```sh
# Use motifs from database to annotate (combined with de novo annotation)
vampire anno -f -t 8 [prefix] [output_prefix]

# Only use motifs from database to annotate (without de novo annotation)
vampire anno -f --no-denovo -t 8 [prefix] [output_prefix]
```
For more detailed instructions and examples, refer to [the VAMPIRE Cookbook](https://zikun-yang.github.io/VAMPIRE_Cookbook/).

### <a name="generator"></a>generator - Generate simulated TR sequences
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

### <a name="mkref"></a>mkref - Create reference motifset

The `mkref` function can generate motif database (in FASTA format) from VAMPIRE annotation results. It can corporate with the `anno` function to annotate TR sequences in a two-step approach: firstly, use `anno` to annotate TR sequences, then use `mkref` to generate motif database from the annotation results. Then, the `anno` function can use this motif database in non-de novo mode to annotate TR sequences.  This two-step approach can generate a motif database on population level and annotate TR sequences with high accuracy.
```sh
# Create reference motif set from annotation results
vampire mkref tests/003-mkref_data tests/003-mkref_reference.fa
```


### <a name="evaluate"></a>evaluate - Evaluate annotation quality
VAMPIRE evaluates the quality of annotation in a edit distance matrix method. See [the VAMPIRE Cookbook](https://zikun-yang.github.io/VAMPIRE_Cookbook/) for more details.
```sh
# Evaluate the quality of annotation
vampire evaluate tests/001-anno_STR tests/004-evaluate
```
Four figures will be generated, combining two modes (`raw` and `normalized`) with strand options (`merge` and `seperate`). For detailed machanisms, usage and interpretation of the `raw` and `normalized` modes as well as the `merge` and `seperate` strand settings, please refer to [the VAMPIRE Cookbook](https://zikun-yang.github.io/VAMPIRE_Cookbook/).

### <a name="refine"></a>refine - Refine annotation

This refinement process will generate a new annotation file with the same format as the input with the refinement action provided by user. Three operations (`MERGE`, `REPLACE` and `DELETE`) are supported.
```sh
# Refine the annotation
vampire refine tests/001-anno_STR tests/005-refine_action.tsv -o tests/005-anno_STR.revised
```

### <a name="logo"></a>logo - Plotting sequence logos to visualize motif variation

VAMPIRE plots sequence logos in three types: count, probability, and information score. By default, VAMPIRE plot sequence logos using the motif statistics file `*.motif.tsv`. If you want to plot sequence logos using the annotation file `*.anno.tsv` to show the true motif variation, use the `--type annotation` option.
```sh
# Plotting sequence logos to visualize motif variation
vampire logo tests/001-anno_STR tests/006-anno_STR_motif
vampire logo --type annotation tests/001-anno_STR tests/006-anno_STR_annotation
```


### <a name="identity"></a>identity - Calculate the identity matrix for TR sequences

VAMPIRE uses alignment-based method to calculate the identity matrix for TR sequences.
```sh
# Calculate the identity matrix for TR sequences
vampire identity -t 20 -w 30 tests/001-anno_STR tests/007-anno_STR
```
By default, VAMPIRE do not account for insertion and deletion events when generating the identity matrix. To include such events within a specific length range, use the `--max-indel` and `--min-indel` options to set the maximum and minimum indel lengths to consider.

After generating the identity matrix, you can visualize the heatmap with repeatmasker annotation and TR strand information using this command:
```sh
python scripts/get_visualization_data.py --prefix [annotation_prefix] --repeat [repeatmasker_annotation] --output [output_prefix]
Rscript scripts/SG_aln_plot.R -t 30 -b [identity_bed_file] -a [visualization_data] -p [figure_output_prefix]
```

## <a name="help"></a>Getting Help

For detailed description of options, please see [VAMPIRE Cookbook](https://zikun-yang.github.io/VAMPIRE_Cookbook/). If you have further questions, want to report a bug, or suggest a new feature, please raise an issue at the [issue page](https://github.com/zikun-yang/VAMPIRE/issues).

<!--
## <a name="limitations"></a>Limitations

- VAMPIRE is designed for annotating the variation of TRs. While it can be used for genome-wide TR annotation with basic information, it can be time-consuming due to the additional data it processes. To address this, we plan to develop a `scan` function optimized for whole-genome TR annotation.
- TRs with very low copy numbers may be challenging to annotate accurately due to the limited availability of k-mers.
-->

## <a name="cite"></a>Citating VAMPIRE

If you use VAMPIRE in your work, please cite:
> To be updated

