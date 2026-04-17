# anno

Annotate a single tandem repeat (TR) locus with motif decomposition and variation analysis.

This subcommand takes a FASTA file containing one or more sequences of a single TR locus (e.g., a centromere or satellite array) and performs:

1. **Decomposition** — Builds a De Bruijn graph from k-mers to discover the underlying motif set, either de novo or using a reference motif database.
2. **Annotation** — Aligns the sequence against the discovered motifs to produce a motif-level annotation (position, copy number, orientation).
3. **Report generation** — Produces tabular outputs and an interactive HTML report summarizing motif composition and variation.

## Usage

```bash
vampire anno [options] <input.fa> <output_prefix>
```

## Examples

```bash
# Auto-detect parameters
vampire anno --auto input.fa output/prefix

# Manual k-mer size with custom score threshold
vampire anno -k 13 --score 15 CEN1.fa output/CEN1

# Use a custom reference motif set
vampire anno -m /path/to/motifs.tsv input.fa output/prefix
```

## Arguments

| Argument | Description |
|----------|-------------|
| `input` | Input FASTA file to annotate |
| `prefix` | Output prefix for all result files |

## Options

### General Options

| Option | Default | Description |
|--------|---------|-------------|
| `-t, --thread, --threads` | `4` | Number of threads |
| `--auto` | `False` | Automatically estimate k-mer size and related parameters |
| `--debug` | `False` | Output debug info and keep temporary files |
| `--seq-win-size` | `5000` | Parallel window size for annotation (bp) |
| `--seq-ovlp-size` | `1000` | Overlap between consecutive windows (bp) |
| `-r, --resource` | `50` | Memory limit (GB) |

### Decomposition Options

| Option | Default | Description |
|--------|---------|-------------|
| `-k, --ksize` | `9` | k-mer size for building the De Bruijn graph |
| `-m, --motif` | `base` | Reference motif set path. Use `base` for the built-in default. |
| `-n, --motifnum` | `30` | Maximum number of motifs to discover |
| `--abud-threshold` | `0.01` | Minimum edge weight threshold relative to the top edge weight in the De Bruijn graph |
| `--abud-min` | `3` | Minimum absolute edge weight in the De Bruijn graph |
| `--no-denovo` | `False` | Skip de novo motif discovery; annotate using only the reference motif set |

### Annotation Options

| Option | Default | Description |
|--------|---------|-------------|
| `-f, --force` | `False` | Force-add reference motifs into the annotation module |
| `--annotation-min-similarity` | `0.6` | Minimum motif similarity required for annotation |
| `--finding-min-similarity` | `0.8` | Minimum motif similarity to match a query against the reference motif set |
| `--match-score` | `2` | Match score for alignment |
| `--mismatch-penalty` | `7` | Mismatch penalty for alignment |
| `--gap-open-penalty` | `7` | Gap open penalty for alignment |
| `--gap-extend-penalty` | `7` | Gap extend penalty for alignment |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-report` | `False` | Skip HTML report generation |
| `-s, --min-score` | `5` | Minimum row score for the concise output table (`*.concise.tsv`) |

## Output Files

Results are written with the provided `<prefix>`:

- `<prefix>.annotation.tsv` — Per-segment annotation (motif, CIGAR, position, sequence)
- `<prefix>.concise.tsv` — Locus-level summary per sample (copy number, motif array, orientation)
- `<prefix>.motif.tsv` — Discovered motif metadata (motif sequence, copy number, label)
- `<prefix>.distance.tsv` — Pairwise motif distance matrix
- `<prefix>.report.html` — Interactive HTML report (unless `--skip-report` is set)
