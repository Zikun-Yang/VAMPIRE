# refine

Refine tandem repeat annotations using a user-defined action file.

This subcommand loads a previously generated annotation and applies a set of manual or scripted corrections (e.g., merging motifs, splitting loci, or adjusting boundaries) specified in an action file.

## Usage

```bash
vampire refine [options] <prefix> <action_file>
```

## Example

```bash
vampire refine results/scan actions.json -o results/scan.revised
```

## Arguments

| Argument | Description |
|----------|-------------|
| `prefix` | Output prefix of the raw annotation results to refine |
| `action` | Action file describing the corrections to apply |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --out` | `<prefix>.revised` | Output prefix for refined results |
| `-t, --thread, --threads` | `8` | Number of threads |

## Output Files

Results are written with the provided output prefix:

- Refined annotation tables and reports after applying the specified actions.
