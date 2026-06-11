# Release Notes

## v0.4.0 (2026-06-04)

### New Features

- **`scan` subcommand** — End-to-end tandem-repeat scanning pipeline with HTML report generation.
- **`anno` module rewrite** — New `pp` / `tl` / `pl` API layered on `anndata`, with job-directory support for reproducible workflows.
- **Haplotype analysis** — Fused kNN graph clustering (`haplotype_neighbor`), Leiden clustering (`haplotype_leiden`), and resolution optimization (`haplotype_optimize_leiden_resolution`) for grouping samples by structural/compositional/CNV similarity.
- **New plotting functions**
  - `tracksplot()` — multi-track genomic visualization (bed, bedgraph, heatmap)
  - `waterfall()` / `waterfall_legend()` — motif composition across samples
  - `logo()` — sequence logos for aligned motifs
  - `haplotype_evaluation()` — resolution-scan evaluation curve
  - `heatmap`, `violin`, `pca`, `sizing` helpers
- **Dataset loaders** — `chm13_cen1_tracks`, `wdr7_hprc`, and other example datasets shipped with the package.

### Infrastructure

- GitHub Actions CI with cross-platform testing (Python 3.10–3.12).
- Full test suite for `anno` workflows.
- Sphinx documentation with tutorials.

### Breaking Changes

- The `anno` command-line interface and Python API have been completely restructured. Scripts written against v0.3.x will need migration.
- `trackplot` renamed to `tracksplot`.
