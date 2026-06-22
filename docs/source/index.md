[![Stars](https://img.shields.io/github/stars/Zikun-Yang/VAMPIRE?style=flat&logo=GitHub&color=yellow)](https://github.com/Zikun-Yang/VAMPIRE/stargazers)
[![PyPI](https://img.shields.io/pypi/v/vampire-tr)](https://pypi.org/project/vampire-tr)
[![PyPI Downloads](https://img.shields.io/pepy/dt/vampire-tr?logo=pypi)](https://pepy.tech/project/vampire-tr)
[![Docs](https://readthedocs.org/projects/vampire-tr/badge/?version=latest)](https://vampire-tr.readthedocs.io/en/latest/index.html)


```{toctree}
:hidden: true
:maxdepth: 1

installation
tutorials/index
cli/index
api/index
release-notes/index
contributing/index
references
```

# VAMPIRE - VAriation and Motif Patterns In tandem REpeats

VAMPIRE is a unified framework for *de novo* tandem repeat (TR) annotation and analysis. It systematically characterizes copy number variation, motif variation and structural variation within TR arrays.

By representing TR arrays as hierarchical motif compositions and quantifying copy-number changes, motif substitutions, and array restructuring across samples, VAMPIRE transforms raw sequence data into standardized, interpretable, and queryable repeat-variation matrices. Through its AnnData-based data model, VAMPIRE enables seamless integration with downstream analysis workflows.

Read [the documentation](https://vampire-tr.readthedocs.io/en/latest/index.html). Open an issue or create a pull request if you would like to contribute.

**Version:** v0.4.2

## Quick Start

Not sure where to begin? Select the workflow that matches your data:

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Annotate multiple TRs on genome
:link: cli/scan
:link-type: doc

*De novo* TR discovery from genome assembly.
:::

:::{grid-item-card} Annotate single TR locus across population
:link: cli/anno
:link-type: doc

TR variation profiling across samples at target loci.
:::

::::

## Walkthrough Analysis

Ready to dive deeper? Follow these step-by-step examples:

### Single locus

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} STR/VNTR analysis across populations
:link: tutorials/basic/wdr7_analysis
:link-type: doc

Example analysis of 69 bp VNTR in the intron of the gene WDR7 across human population.
:::

:::{grid-item-card} Centromere and satellites analysis and visualization
:link: tutorials/basic/cen1_analysis
:link-type: doc

Example analysis and visualization of the centromere region of T2T-CHM13v2.0 chromosome 1.
:::

::::

## News

```{include} news.md
:start-after: '<!-- marker: after prelude -->'
:end-before: '<!-- marker: before old news -->'
```

