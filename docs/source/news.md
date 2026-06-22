# News

<!-- marker: after prelude -->

#### 2026-06-22 - VAMPIRE v0.4.2 Release

- Fixed a bug in `src/vampire/_anno.py`.
- Added GitHub Actions workflow to build and push Docker images to `zikunyang/vampire-tr` automatically on version tags.

<!-- marker: before old news -->

#### 2026-06-22 - VAMPIRE v0.4.1 Release

- Fixed missing runtime dependencies: `parasail>=1.3`, `nbformat>=4.2.0`, `ipywidgets>=7.0`.
- These packages are now declared in `pyproject.toml` so `pip install vampire-tr` brings in everything needed for CLI and Jupyter notebook usage.

#### 2026-06-22 - VAMPIRE v0.4.0 Release

- Added API framework (`vp.datasets`, `vp.anno.pp`, `vp.anno.pl`, `vp.anno.tl`) (Issues [#2](https://github.com/Zikun-Yang/VAMPIRE/issues/2), [#3](https://github.com/Zikun-Yang/VAMPIRE/issues/3))
- Added CLI (`scan`)
- Added `vp.datasets` module: `wdr7_hprc()`, `ancestry()`
- We update `anno` with length-restrained loop searching method to resolve more complex tandem repeats.
- We add the parameter `--use-raw` to `anno`. This can preserve all different motifs without compression. 

#### 2026-05-19 — VAMPIRE v0.4.0a2 (preview)

Second alpha release with bug fixes, CI improvements, and relaxed integration tests.

- **Bug fix:** Fixed motif polishing logic that lost unmatched de-novo motifs.
- **CI:** Removed ruff lint workflow; relaxed exact DataFrame comparison in integration tests.
- **Tests:** Added `--job` parameter for subprocess isolation between `scan` and `anno`.

#### 2026-05-07 — VAMPIRE v0.4.0a1 (preview)

Preview release of VAMPIRE 0.4.0 with a refactored `scan` / `anno` module split, polars 1.x compatibility, and updated container definition.

- **Breaking change:** `vampire.scan` and `vampire.anno` are now separate top-level modules with their own `pp`, `pl`, and `tl` subpackages.
- **Dependencies:** Removed unused packages (`Levenshtein`, `rich`, `logomaker`). Added minimum version constraints. Upgraded to `polars>=1.0`.
- **Container:** `vampire.def` now supports conditional Tsinghua TUNA mirrors for Chinese users and no longer bundles R.