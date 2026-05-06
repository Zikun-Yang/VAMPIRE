# integrate

Integrate tandem repeat annotations across multiple samples.

This subcommand merges TR annotations from different samples into a unified reference frame. It aligns TR sequences across samples using minimap2, anchors equivalent loci, and produces a consensus annotation table suitable for downstream cross-sample comparison.

## Usage

```bash
vampire integrate [options] <input.tsv> <output_prefix>
```

## Example

```bash
# Use the first sample as the reference for anchoring
vampire integrate --reference samples.tsv results/integrated
```

The input TSV file should contain three columns (no header):

| Column | Description |
|--------|-------------|
| 1 | Sample name |
| 2 | Path to the genome FASTA file |
| 3 | Path to the annotation BED file produced by `vampire scan` |

## Arguments

| Argument | Description |
|----------|-------------|
| `input` | Input TSV file listing samples, genomes, and annotations |
| `prefix` | Output prefix for integrated results |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-r, --reference` | `False` | Use the first sample as the reference for cross-sample anchoring |
| `-t, --thread, --threads` | `16` | Number of threads |
| `-a, --alignment-params` | `-x asm20 --secondary=no --cs` | Alignment parameters passed to minimap2 |
| `-f, --flanking-length` | `100` | Flanking sequence length (bp) to extract around each TR locus |
| `--redo` | `False` | Overwrite existing intermediate results |
| `--debug` | `False` | Output debug info and keep temporary files |

## Output Files

Results are written with the provided `<prefix>`:

- `<prefix>.integrated.tsv` — Merged cross-sample annotation table
