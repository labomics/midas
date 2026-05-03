"""Bundled example datasets shipped inside the scmidas wheel.

These are toy-sized subsets of real datasets, designed to make the README
quickstart runnable in under a minute on a single GPU. They are NOT meant
for benchmarking — see the basics tutorials for full-size data.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import mudata as mu


def quickstart_path() -> Path:
    """Return the on-disk path of the bundled quickstart .h5mu file.

    Returns:
        Path:
            Absolute path to ``quickstart_pbmc_mosaic.h5mu`` inside the
            installed scmidas package.
    """
    with resources.as_file(
        resources.files('scmidas').joinpath('data/quickstart_pbmc_mosaic.h5mu')
    ) as p:
        return Path(p)


def quickstart() -> mu.MuData:
    """Load the bundled quickstart MuData (PBMC RNA+ADT mosaic, 1600 cells).

    The dataset is a hand-tuned subset of the WNN PBMC mosaic dataset:
    4 batches × 400 cells each (RNA-only, ADT-only, two paired) with
    500 RNA HVGs + 224 ADT features, sized so that
    ``scmidas.integrate(...)`` finishes in roughly one minute on a
    single mid-range GPU. **It is intended for the quickstart only**;
    its size and feature count are not appropriate for serious analysis.

    Returns:
        MuData:
            A MuData with two modalities (``'rna'``, ``'adt'``) and the
            following ``obs`` columns at top level:
            ``'batch'`` and ``'celltype'``.
    """
    return mu.read_h5mu(str(quickstart_path()))
