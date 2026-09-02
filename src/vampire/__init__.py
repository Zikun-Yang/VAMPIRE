import os

# Polars uses a Rust Rayon thread pool that is not fork-safe. Vampire uses
# multiprocessing.Pool (fork-based) in several places, including the DP step in
# _anno.py. If polars is imported with its default multi-threaded pool, forked
# child processes can deadlock when they later create a DataFrame. Setting this
# before any submodule import ensures polars sees it at initialization time.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from . import scan
from . import anno
from . import datasets

__all__ = [
    "scan",
    "anno",
    "datasets",
]
