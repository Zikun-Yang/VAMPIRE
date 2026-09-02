---
tocdepth: 0
---

(release-notes)=

# Release notes

## Future features

| Status | Feature Description | Priority | Target Version |
|:---|:---|:---:|:---:|
| Planned   | Comprehensive metadata of human populational genome resources | P1 | v0.4 |
| Planned   | Downstream analysis module for `scan` | P1 | v0.5 |
| Discussed | Disease-associated expansion quantification | P3 | v0.5-0.6 |
| Discussed | TR catalog generation pipeline | P3 | v0.5-0.6 |

## Versions

### **[Latest] — v0.4.3 (2026-07-14)**

#### New Features
- Added automatic k-size selection for `anno`. The tool runs a quick internal `scan`, estimates the motif period range, and picks the k-size whose distance-derived periodicity best matches the expected period.
- Added adaptive boundary padding in `anno` so that boundary k-mers appear in multiple copies and can form SCCs in the De Bruijn graph.
- Added exact motif occurrence counting with an Aho-Corasick automaton; `anno` now reports true copy numbers across all input sequences rather than De Bruijn graph edge-weight estimates.
- Added `--use-raw` output mode for `anno` to export observed aligned sequences as raw motifs without polishing or compression.

#### Improvements
- Set `POLARS_MAX_THREADS=1` in `vampire/__init__.py` before importing any submodule to avoid fork-related deadlocks when `anno` uses `multiprocessing.Pool`.
- Updated CLI help examples for `vampire anno` and `vampire generator`.

#### Bug Fixes
- Fixed a bug in `src/vampire/_anno.py`.

### v0.4.2 (2026-06-22)

#### Bug Fixes
- Fixed a bug in `src/vampire/_anno.py`.

#### New Features
- Added GitHub Actions workflow (`.github/workflows/docker.yml`) to build and push Docker images to Docker Hub (`zikunyang/vampire-tr`) on every `v*` tag push.
- Added `Dockerfile` at repository root.

### v0.4.1 (2026-06-22)

#### Bug Fixes
- Added missing runtime dependencies discovered during v0.4.0 usage:
  - `parasail>=1.3` (required by `vampire.anno.tl.motif_msa`)
  - `nbformat>=4.2.0` (required by Plotly for Jupyter rendering)
  - `ipywidgets>=7.0` (required by `tqdm` for Jupyter progress bar widgets)

### v0.4.0 (2026-06-22)

#### New Features
- Added API framework (`vp.datasets`, `vp.anno.pp`, `vp.anno.pl`, `vp.anno.tl`) (Issues [#2](https://github.com/Zikun-Yang/VAMPIRE/issues/2), [#3](https://github.com/Zikun-Yang/VAMPIRE/issues/3))
- Added CLI (`scan`)
- Added `vp.datasets` module: `wdr7_hprc()`, `ancestry()`

#### Improvements
- We update `anno` with length-restrained loop searching method to resolve more complex tandem repeats.
- We add the parameter `--use-raw` to `anno`. This can preserve all different motifs without compression. 

#### Bug Fixes
- Fixed Issue [#5](https://github.com/Zikun-Yang/VAMPIRE/issues/5)

### **[Initial] — v0.3.0 (2025-06-02)**

#### New Features
- Added CLI framework (`anno`, ...)
