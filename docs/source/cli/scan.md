# scan

Scan and annotate tandem repeats (TRs) across a genome or long sequences.

This subcommand performs a genome-wide (or sequence-wide) scan to detect tandem repeat regions. It uses a multi-scale k-mer smoothness approach to identify candidate TR loci, followed by banded dynamic programming alignment to annotate period, copy number, and motif composition for each locus.

## Usage

```bash
vampire scan [options] <input.fa> <output_prefix>
```

## Example

```bash
# Basic scan with default parameters
vampire scan genome.fa results/genome_scan

# Scan with more threads and custom window size
vampire scan -t 16 --seq-win-size 10000000 genome.fa results/scan
```

## Arguments

| Argument | Description |
|----------|-------------|
| `input` | Input FASTA file to scan for tandem repeats |
| `prefix` | Output prefix for all result files |

## Options

### General Options

| Option | Default | Description |
|--------|---------|-------------|
| `-t, --thread, --threads` | `8` | Number of threads |
| `--debug` | `False` | Output debug info and keep temporary files |
| `--seq-win-size` | `5000000` | Window sequence size for scanning (bp) |
| `--seq-ovlp-size` | `100000` | Overlap sequence size between consecutive windows (bp) |

### Candidate Finding Options

| Option | Default | Description |
|--------|---------|-------------|
| `--ksize` | `17,13,9,5,3` | Comma-separated list of k-mer sizes for candidate detection. Smaller k-mer sizes improve sensitivity for long-period TRs. |
| `--rolling-win-size` | `5` | Rolling window size to compute smoothness score |
| `--min-smoothness` | `50` | Minimum smoothness score to call a candidate region |

### Alignment Options

| Option | Default | Description |
|--------|---------|-------------|
| `--match-score` | `2` | Match score for alignment |
| `--mismatch-penalty` | `7` | Mismatch penalty for alignment |
| `--gap-open-penalty` | `7` | Gap open penalty for alignment |
| `--gap-extend-penalty` | `7` | Gap extend penalty for alignment |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --max-period` | `1000` | Maximum period (motif length) for output |
| `-s, --min-score` | `50` | Minimum alignment score for output |
| `-c, --min-copy` | `1.5` | Minimum copy number for output |
| `--secondary` | `1.0` | Minimum secondary annotation score relative to primary. Set to `1.0` to disable secondary annotation. |
| `--format` | `trf` | Output format (`brief`, `trf`, or `bed`) |
| `--skip-cigar` | `False` | Skip CIGAR string in output |
| `--skip-report` | `False` | Skip HTML report generation |

## Output Files

Results are written with the provided `<prefix>`:

- `<prefix>.trf.tsv` — Main annotation table in TRF-like format
- `<prefix>.report.html` — Interactive HTML report (unless `--skip-report` is set)
