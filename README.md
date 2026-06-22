# VAMPIRE - VAriation and Motif Patterns In tandem REpeats
[![Stars](https://img.shields.io/github/stars/Zikun-Yang/VAMPIRE?style=flat&logo=GitHub&color=yellow)](https://github.com/Zikun-Yang/VAMPIRE/stargazers)
[![PyPI](https://img.shields.io/pypi/v/vampire-tr)](https://pypi.org/project/vampire-tr)
[![PyPI version](https://badge.fury.io/py/vampire-tr.svg)](https://pypi.org/project/vampire-tr/)  [![Docker Image Version](https://img.shields.io/badge/docker-v0.3.0-blue)](https://hub.docker.com/r/zikunyang/vampire-tr/tags) [![License](https://img.shields.io/github/license/Zikun-Yang/VAMPIRE)](https://github.com/Zikun-Yang/VAMPIRE/blob/main/LICENSE)  [![Last Commit](https://img.shields.io/github/last-commit/Zikun-Yang/VAMPIRE)](https://github.com/Zikun-Yang/VAMPIRE/commits/main)
[![PyPI Downloads](https://img.shields.io/pepy/dt/vampire-tr?logo=pypi)](https://pepy.tech/project/vampire-tr)
[![Docs](https://readthedocs.org/projects/vampire-tr/badge/?version=latest)](https://vampire-tr.readthedocs.io/en/latest/index.html)

## <a name="started"></a>Getting Started
```sh
# Install
mamba create -n vampire python=3.10 -y
mamba activate vampire
mamba install vampire-tr 

# Annotate TRs on genomes
vampire scan <fasta> <prefix>

# Annotate single TR locus across population
vampire anno <fasta> <prefix>

# Generate simulated TR sequences
vampire generator -m GGC -l 1000 -r 0.01 -p <prefix>
vampire generator -m GGC GGT -l 1000 -r 0.01 -p <prefix>

# Calculate the identity matrix for TR sequences
vampire identity -w 5 <anno_prefix> <identity_prefix>
```
See [Docs](https://vampire-tr.readthedocs.io/en/latest/index.html) for more details.

## <a name="toc"></a>Table of Contents

- [Getting Started](#started)
- [Introduction](#intro)
- [Why VAMPIRE?](#why)
- [Installation](#install)
- [Usage](#usage)
    - [Annotate TRs on genome](#scan)
    - [Annotate single TR locus across population](#anno)
    - [Generate simulated TR sequences](#generator)
    - [Calculate identity matrix](#identity)
- [Getting Help](#help)
- [Citing VAMPIRE](#cite)

## <a name="intro"></a>Introduction

VAMPIRE is a unified framework for *de novo* [tandem repeat (TR)](https://en.wikipedia.org/wiki/Tandem_repeat) annotation and analysis. It systematically characterizes copy number variation, motif variation and structural variation within TR arrays.

By representing TR arrays as hierarchical motif compositions and quantifying copy-number changes, motif substitutions, and array restructuring across samples, VAMPIRE transforms raw sequence data into standardized, interpretable, and queryable repeat-variation matrices. Through its AnnData-based data model, VAMPIRE enables seamless integration with downstream analysis workflows.

Read [the documentation](https://vampire-tr.readthedocs.io/en/latest/index.html). Open an issue or create a pull request if you would like to contribute.

## <a name="why"></a>Why VAMPIRE?

- **Beyond Copy Number:** VAMPIRE uncovers not only copy number but also internal variation.
- **Flexible and Comprehensive:** Its customizable parameters support the annotation of a wide range of TRs, from short tandem repeats (STRs) and variable number tandem repeats (VNTRs) to megabase-scale satellite arrays.
- **Analysis Ecosytem:** VAMPIRE contains `vp.anno.pp`, `vp.anno.pl`, `vp.anno.tl` modules for analysis and plotting.

## <a name="install"></a>Installation

```sh
# Install by pip 
mamba create -n vampire python=3.10 -y
mamba activate vampire
mamba install vampire-tr 
```
    
## <a name="usage"></a>Usage

VAMPIRE contains several subcommands. Here we list `scan`, `anno`, `generator` and `identity`.

### <a name="scan"></a>scan - Annotate TRs on genome

VAMPIRE can scan genome assemblies or long sequences to detect tandem repeat (TR) loci. It uses a multi-scale k-mer smoothness approach to identify candidate regions, followed by banded alignment to annotate period and copy number for each locus.

```sh
# Scan a genome with 8 threads
vampire scan -t 8 genome.fa genome_scan

# Output results in BED format
vampire scan --format bed genome.fa genome_scan
```

### <a name="anno"></a>anno - Annotate single TR locus across population

One of the primary uses of VAMPIRE is to annotate tandem repeat (TR) sequences from input files in FASTA format. A typical command is as follows:
```sh
# de novo annotate TR sequences
vampire anno -t 8 <fasta> <prefix>
```
where `-t` sets the number of threads, `tests/001-anno_STR.fa` is the input sequences, and `tests/001-anno_STR` is the output prefix. By default, VAMPIRE use the built-in `base` motif database to refine and label motifs. This database includes pCht/StSat in *Pan* and human alpha-satellite mononers from the paper:
> Altemose N, Logsdon G A, Bzikadze A V, et al. 
> Complete genomic and epigenetic maps of human centromeres[J]. 
> Science, 2022, 376(6588): eabl4178.

For more detailed instructions and examples, refer to [the VAMPIRE Docs](https://vampire-tr.readthedocs.io/en/latest/index.html).

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

### <a name="identity"></a>identity - Calculate the identity matrix for TR sequences

VAMPIRE uses alignment-based method to calculate the identity matrix for TR sequences.
```sh
# Calculate the identity matrix for TR sequences
vampire identity -t 20 -w 30 <anno_prefix> <identity_prefix>
```
By default, VAMPIRE do not account for insertion and deletion events when generating the identity matrix. To include such events within a specific length range, use the `--max-indel` and `--min-indel` options to set the maximum and minimum indel lengths to consider.

After generating the identity matrix, you can visualize the identity heatmap using `vp.anno.pl.tracksplot()`.

## <a name="help"></a>Getting Help

For detailed description of options, please see [VAMPIRE Docs](https://vampire-tr.readthedocs.io/en/latest/index.html). If you have further questions, want to report a bug, or suggest a new feature, please raise an issue at the [issue page](https://github.com/zikun-yang/VAMPIRE/issues).

<!--
## <a name="limitations"></a>Limitations

- VAMPIRE is designed for annotating the variation of TRs. While it can be used for genome-wide TR annotation with basic information, it can be time-consuming due to the additional data it processes. To address this, we plan to develop a `scan` function optimized for whole-genome TR annotation.
- TRs with very low copy numbers may be challenging to annotate accurately due to the limited availability of k-mers.
-->

## <a name="cite"></a>Citating VAMPIRE

If you use VAMPIRE in your work, please cite:
> To be updated