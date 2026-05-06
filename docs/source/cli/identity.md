# identity

Calculate sequence identity within tandem repeat regions.

This subcommand computes local sequence identity across TR loci using a sliding motif-level window. It is useful for assessing the purity or divergence of repeat arrays.

## Usage

```bash
vampire identity [options] <input_prefix> <output_prefix>
```

## Example

```bash
# Compute identity with default settings
vampire identity results/scan results/identity

# Use a larger window and allow small indels
vampire identity -w 200 --max-indel 2 results/scan results/identity
```

## Arguments

| Argument | Description |
|----------|-------------|
| `prefix` | Prefix of the input annotation files |
| `output` | Output prefix for identity results |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-w, --window-size` | `100` | Sliding window size (in motifs) |
| `-t, --thread, --threads` | `30` | Number of threads |
| `--mode` | `raw` | Analysis mode: `raw` or `invert` |
| `--max-indel` | `0` | Maximum allowed indel length |
| `--min-indel` | `0` | Minimum indel length to consider |

## Output Files

Results are written with the provided `<output>` prefix:

- Identity tables and visualizations showing local divergence across repeat arrays.
