---
title: FAQs
layout: default
nav_order: 60
---

# **FAQs**

Have a question that is not listed here? [File an issue.](https://github.com/Zikun-Yang/VAMPIRE/issues/new?template=bug_report.md)

## ❓**[anno] What kind of input does VAMPIRE accept?**
VAMPIRE takes a FASTA file as input. Multiple sequences in the FASTA will process independently. 

## ❓**[anno] Why is my output empty?**
Make sure your input file is in proper FASTA format and is not empty.

Ensure that the input file actually contains tandem repeats. If the tamdem repeat has too few copy number or high motif variation, VAMPIRE can ignore the tandem repeats. Try to reduce the minimum kmer number filter `--abud-min` and `--abud-threshold`. You can also configure a consensus motif with `--motif consensus.fa --no-denovo` to annotate. 

If you're using custom parameters, try running with the default settings first ot try a looser parameter set.

## ❓**[anno] Can I run VAMPIRE on large genomes?**
Yes, but performance depends on genome size and repeat density. Consider splitting your input with `--window-length 500 --overlap-length 100`.

A new feature `scan` is considered to be added into further version, to support rapid TR annotation on whole genome sequence with varaint motif calling.

## ❓**[logo] Why are the base counts different across loci?** 
This is because, during motif alignment processing, VAMPIRE ignores the continuous 5' or 3' deleted part of a incomplete motif. For example, a aligned motif `----ATCAA-TGGC`, the first four deletion `-` will be ignored but the deletion `-` inside will be counted.

This behavior is intentional due to the existence of imcomplete motifs at the boundaries of TR sequences. By excluding the continuous missing sequence at 5' or 3' end, VAMPIRE ensures that the logo reflects the true deletion.
