# Installation

::::{tab-set}

:::{tab-item} Singularity (Recommended)
:sync: singularity

Pull the prebuilt Singularity image:

```console
$ singularity pull docker://zikunyang/vampire-tr:latest
$ singularity exec vampire-tr_latest.sif vampire --help
```

Build from definition file:

```console
$ git clone https://github.com/Zikun-Yang/VAMPIRE.git
$ cd VAMPIRE
$ singularity build vampire-tr_latest.sif vampire.def
$ singularity exec vampire-tr_latest.sif vampire --help
```
:::

:::{tab-item} Pip
:sync: pip

Install VAMPIRE using pip:

```console
$ pip install vampire-tr
```
:::

:::{tab-item} Docker
:sync: docker

```console
$ docker pull zikunyang/vampire-tr:latest
$ docker run -it --name vampire-tr zikunyang/vampire-tr:latest
$ docker exec vampire-tr vampire --help
```
:::

::::

## Requirements

VAMPIRE requires Python 3.10+ to run. Core Python dependencies (including `polars`, `numpy`, `plotly`, `anndata`, `biopython`, `numba`, `pysam`, etc.) are installed automatically when you install the package via `pip`. Some external software is required for specific functions; for example, `mafft` is needed for the sequence-logo plotting feature.

## Verification

After installation, verify that VAMPIRE is installed correctly:

```console
$ vampire --help
```
You should see the VAMPIRE help message with available commands.
