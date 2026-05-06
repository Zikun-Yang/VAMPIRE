# generator

Generate simulated tandem repeat sequences from reference motifs.

This subcommand creates synthetic tandem repeat sequences by tiling one or more input motifs to a target length, optionally introducing random substitutions at a specified mutation rate. It is useful for benchmarking, testing downstream analysis pipelines, and generating controlled synthetic data.

## Usage

```bash
vampire generator -m <motif> ... -p <prefix> [options]
```

## Examples

```bash
# Generate a pure tandem repeat from a single motif
vampire generator -m GGC -l 1000 -r 0 -p output/prefix

# Generate a mixed tandem repeat from multiple motifs with 5% mutation
vampire generator -m GGC GGT -l 1000 -r 0.05 -p output/prefix
```

## Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `-m, --motifs` | **Yes** | — | One or more reference motifs (e.g., `GGC` or `GGC GGT`) |
| `-p, --prefix` | **Yes** | — | Output prefix for generated files |
| `-l, --length` | No | `1000` | Total length of the simulated tandem repeat sequence (bp) |
| `-r, --mutation-rate` | No | `0` | Substitution mutation rate, between `0` and `1` |
| `-s, --seed` | No | `42` | Random seed for reproducibility |
| `--debug` | No | `False` | Output debug information |

## Output Files

Results are written with the provided `<prefix>`:
- `<prefix>.fa` — Generated tandem repeat sequence
- `<prefix>.anno.tsv` — mutation-aware motif annotation file
- `<prefix>.anno_woMut.tsv` — motif annotation without mutation

## Notes

- Motifs are tiled randomly to reach the target sequence length.
- Mutations are applied uniformly across the entire sequence after tiling.
- The output includes a FASTA file containing the generated sequence.
