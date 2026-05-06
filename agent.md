# VAMPIRE Agent Notes

## Project Overview

**VAMPIRE** (vampire-tr on PyPI) is a Python package for *de novo* tandem repeat (TR) motif annotation, structural decomposition, and variation profiling. It is developed by Zikun Yang and team, hosted at `https://github.com/Zikun-Yang/VAMPIRE`.

- **Version:** 0.3.0
- **License:** MIT
- **Python requirement:** `>=3.10,<3.13`
- **Package name on PyPI:** `vampire-tr`
- **Entry point:** `vampire = vampire.main:main`

## Repository Structure

```
VAMPIRE/
├── src/vampire/           # Main Python package
│   ├── __init__.py        # Exports pp, pl, tl
│   ├── main.py            # CLI entry point
│   ├── anno.py            # Core annotation logic
│   ├── scan.py            # Genome-wide TR scanning
│   ├── generator.py       # Simulated TR sequence generation
│   ├── evaluate.py        # Annotation quality evaluation
│   ├── refine.py          # Annotation refinement
│   ├── identity.py        # Identity matrix calculation
│   ├── integrate.py       # Integration utilities
│   ├── logo.py            # Sequence logo plotting
│   ├── mkref.py           # Reference motif set creation
│   ├── estimateParameters.py
│   ├── _utils.py          # Internal utilities (numba helpers)
│   ├── pp/                # Preprocessing (public API)
│   │   ├── __init__.py
│   │   └── _read.py       # read_bed, read_bedgraph, read_indexed_bed, read_anno
│   ├── pl/                # Plotting (public API)
│   │   ├── __init__.py
│   │   ├── _plot.py       # trackplot
│   │   └── _setting.py    # set_default_plotstyle
│   ├── tl/                # Tools (placeholder, currently empty public API)
│   │   └── __init__.py
│   └── _report_utils/     # Internal report generation helpers
├── docs/                  # Sphinx + MyST documentation
│   ├── source/
│   │   ├── conf.py        # Sphinx config (uses scanpydoc theme)
│   │   ├── index.md
│   │   ├── installation.md
│   │   ├── api/
│   │   │   ├── index.md
│   │   │   ├── pp/index.md
│   │   │   ├── pl/index.md
│   │   │   └── generated/ # autosummary .rst outputs
│   │   └── ...
│   └── requirements.txt   # Doc build deps
├── pyproject.toml         # Package metadata and dependencies
├── readthedocs.yml        # RTD build config
├── vampire.def            # Singularity definition file
├── README.md
└── benchmark/
```

## Public API Modules

The package follows a scanpy-like API convention with `pp`, `pl`, and `tl` submodules.

### `vampire.pp` — Preprocessing

Exported functions in `src/vampire/pp/__init__.py`:

- `read_bed(bed_file, columns=BED_COLS) -> pl.LazyFrame`
- `read_bedgraph(bedgraph_file, columns=BEDGRAPH_COLS) -> pl.LazyFrame`
- `read_indexed_bed(bed_file, chrom, start, end, columns=BED_COLS) -> pl.LazyFrame`
- `read_anno(file, use_raw=False) -> ad.AnnData`

All implemented in `src/vampire/pp/_read.py`.

### `vampire.pl` — Plotting

Exported functions in `src/vampire/pl/__init__.py`:

- `trackplot(tracks, region, title, figsize, vertical_spacing) -> go.Figure`
- `set_default_plotstyle(font_size, font_family, width, height, showgrid)`

Implemented in `src/vampire/pl/_plot.py` and `src/vampire/pl/_setting.py`.

### `vampire.tl` — Tools

Currently empty; `__init__.py` has no exports.

## Documentation Conventions

- The docs use **Sphinx** with **MyST Parser** (`myst_parser`) for Markdown support.
- API docs are generated via `sphinx.ext.autosummary`.
- Public API pages live under `docs/source/api/{pp,pl}/index.md` and use `.. autosummary::` blocks pointing to `../generated/`.
- The HTML theme is `scanpydoc` (configured in `docs/source/conf.py`).
- When adding a new public function to the API docs, add it to the `autosummary` table in the corresponding `index.md` and ensure a `.rst` stub is generated (or manually create it in `docs/source/api/generated/`).

## Key Dependencies

Core runtime dependencies heavily used throughout the codebase but historically incomplete in `pyproject.toml`:

- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `polars` — used extensively for data I/O
- `pysam` — indexed BED reading
- `pyarrow` — backend for Polars / AnnData
- `anndata` — `read_anno` returns `AnnData`
- `plotly` — all plotting in `pl` and report utils
- `numba` — accelerated alignment and utility functions
- `edlib`, `Levenshtein`, `pybktree` — sequence distance operations
- `biopython` (`Bio`) — FASTA I/O
- `sourmash`, `networkx`, `tqdm`, `rich`, `logomaker`, `matplotlib`, `seaborn`
- `resource` — **not a PyPI package**; it is part of the Python standard library on Unix. Do **not** list it in `pyproject.toml` dependencies.

## Environment / Build Notes

- `pyproject.toml` package-data references `"vampire.resources"` with `scan_web_summary_template.html` and `refMotif.fa`. The actual file in the repo is `anno_web_summary_template.html` — the package-data entry is likely stale.
- `docs/source/conf.py` has duplicate entries for `sphinx_autodoc_typehints` and `sphinx_design`.
- `docs/requirements.txt` includes `sphinx-book-theme` and `scanpydoc`, but the `pyproject.toml` `[project.optional-dependencies] docs` section does not include them (it lists `sphinx-rtd-theme` instead, which is unused).
- ReadTheDocs builds use `ubuntu-22.04` + Python 3.11.
- Singularity image is based on `ubuntu:22.04`, installs Miniconda, R 4.4.2, mafft, and then `pip install vampire-tr`.

## CLI Commands

The package exposes 7 subcommands via `vampire`:

- `anno` — annotate TR sequences
- `generator` — generate simulated TR sequences
- `mkref` — create reference motifset
- `evaluate` — evaluate annotation quality
- `refine` — refine annotation
- `logo` — plot sequence logos
- `identity` — calculate identity matrix

## Common Tasks for Agents

1. **Adding new public API functions to docs:**
   - Add to `src/vampire/{pp,pl,tl}/__init__.py` `__all__` if not already there.
   - Add entry to `docs/source/api/{pp,pl}/index.md` in the `autosummary` block.
   - Create `docs/source/api/generated/vampire.{mod}.{func}.rst` if autosummary does not auto-generate it.

2. **Fixing dependencies:**
   - Sync `pyproject.toml` `[project] dependencies` with actual imports in `src/vampire/`.
   - Remove non-PyPI packages like `resource`.
   - Keep `docs/requirements.txt` and `pyproject.toml` `[project.optional-dependencies] docs` in sync.

3. **Updating installation instructions:**
   - Edit `docs/source/installation.md` for docs-site instructions.
   - Edit `README.md` for repo-front-page instructions.
