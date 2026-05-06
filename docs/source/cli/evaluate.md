# evaluate

Evaluate tandem repeat annotation accuracy against a reference.

This subcommand compares annotated motifs (or loci) against a ground-truth reference to compute accuracy metrics. It is useful for benchmarking VAMPIRE on simulated or manually curated datasets.

## Usage

```bash
vampire evaluate [options] <input_prefix> <output_prefix>
```

## Example

```bash
vampire evaluate -t 6 results/scan results/eval
```

## Arguments

| Argument | Description |
|----------|-------------|
| `prefix` | Input prefix of raw annotation results |
| `output` | Output prefix for evaluation results |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-t, --thread, --threads` | `6` | Number of threads |
| `-p, --percentage` | `75` | Threshold percentile for flagging abnormal values (0–100) |
| `-s, --show-distance` | `False` | Show detailed edit distances on the heatmap |

## Output Files

Results are written with the provided `<output>` prefix:

- Evaluation tables and distance heatmaps comparing annotated vs. reference motifs.
