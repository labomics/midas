"""Bundled example datasets shipped inside the scmidas wheel.

These are toy-sized subsets of real datasets, designed to make the README
quickstart runnable in under a minute on a single GPU. They are NOT meant
for benchmarking — see the basics tutorials for full-size data.
"""
from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Optional, Union

import anndata as ad
import mudata as mu

logger = logging.getLogger(__name__)


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


def from_dir(
    dir_path: Union[str, Path],
    label_dir: Optional[Union[str, Path]] = None,
    label_col: str = 'label',
) -> mu.MuData:
    """Load a MIDAS directory-format dataset as a :class:`MuData`.

    The directory format (used by the basics tutorials) lays each batch's
    counts out as MatrixMarket .mtx files plus per-feature mask CSVs::

        dir_path/
            feat/feat_dims.toml          # per-modality chunk sizes
            <batch>/
                cell_names.csv           # cell IDs (1 column, no header beyond default)
                mat/<modality>.mtx       # (n_cells, n_features), Matrix Market
                mask/<modality>.csv      # 1-row CSV, n_features columns (0/1 mask)
            ...

    The returned MuData has:

    - One modality per modality file present in ``feat/feat_dims.toml``.
    - ``mdata[m].obs['batch']`` set to the source batch name.
    - ``mdata[m].uns[f'mask_{batch}']`` for any per-batch feature masks
      that exist (matches the lookup in ``MIDAS.get_info_from_mdata``).
    - ``mdata.uns['feat_dims']`` mirroring ``feat_dims.toml`` so callers
      can pass ``dims_x=mdata.uns['feat_dims']`` to ``setup_mudata``
      (needed for ATAC chromosome chunking).
    - If ``label_dir`` is given, ``mdata[m].obs[label_col]`` is filled in
      from ``label_dir/<batch>.csv`` (matched positionally to cells in
      that batch).

    Parameters:
        dir_path : str or Path
            Path to the ``data/`` directory described above.
        label_dir : str or Path, optional
            Path to the sibling ``label/`` directory; one CSV per batch.
        label_col : str
            Name of the obs column to write labels under.

    Returns:
        MuData: One AnnData per modality, indexed by batch.

    Examples:
        >>> import scmidas
        >>> mdata = scmidas.datasets.from_dir(
        ...     'dataset/teadog_mosaic_mtx/data',
        ...     label_dir='dataset/teadog_mosaic_mtx/label',
        ... )
        >>> scmidas.MIDAS.setup_mudata(mdata, dims_x=mdata.uns['feat_dims'])
        >>> model = scmidas.MIDAS(mdata)
    """
    import natsort
    import numpy as np
    import pandas as pd
    import scipy.io
    import scipy.sparse as sp
    import toml

    p = Path(dir_path)
    if not (p / 'feat' / 'feat_dims.toml').exists():
        raise FileNotFoundError(
            f"{p} doesn't look like a MIDAS dataset dir "
            f"(missing feat/feat_dims.toml)."
        )
    feat_dims = toml.load(p / 'feat' / 'feat_dims.toml')
    modalities = list(feat_dims.keys())

    batch_dirs = [
        d for d in natsort.natsorted([d for d in p.iterdir() if d.is_dir()])
        if d.name != 'feat'
    ]
    if not batch_dirs:
        raise ValueError(f"No batch directories found under {p}.")

    per_mod = {m: {'mats': [], 'cells': [], 'batches': [], 'masks': {}} for m in modalities}

    for bd in batch_dirs:
        b = bd.name
        cn_path = bd / 'cell_names.csv'
        explicit_cell_names = None
        if cn_path.exists():
            explicit_cell_names = pd.read_csv(cn_path, index_col=0).iloc[:, 0].astype(str).tolist()
        mat_dir = bd / 'mat'
        mask_dir = bd / 'mask'

        for m in modalities:
            mat_file = mat_dir / f'{m}.mtx'
            if not mat_file.exists():
                continue
            mat = scipy.io.mmread(str(mat_file)).tocsr()
            n = mat.shape[0]
            if explicit_cell_names is not None:
                if n != len(explicit_cell_names):
                    raise ValueError(
                        f"{mat_file}: rows={n} but {cn_path} has {len(explicit_cell_names)} cells."
                    )
                cell_names = explicit_cell_names
            else:
                cell_names = [f'{b}_{i}' for i in range(n)]
            per_mod[m]['mats'].append(mat)
            per_mod[m]['cells'].append(cell_names)
            per_mod[m]['batches'].append(b)

            mask_file = mask_dir / f'{m}.csv'
            if mask_file.exists():
                mask_arr = pd.read_csv(mask_file, index_col=0).values.flatten().astype(np.float32)
                per_mod[m]['masks'][b] = mask_arr

    mdict = {}
    for m, info in per_mod.items():
        if not info['mats']:
            logger.info("Modality %r had no matrices in any batch; skipping.", m)
            continue
        X = sp.vstack(info['mats']).tocsr()
        all_cells: list = []
        all_batches: list = []
        for cells, batch in zip(info['cells'], info['batches']):
            all_cells.extend(cells)
            all_batches.extend([batch] * len(cells))
        adata = ad.AnnData(X=X)
        adata.obs_names = all_cells
        adata.obs_names_make_unique()
        adata.obs['batch'] = pd.Categorical(all_batches)
        for b, mask in info['masks'].items():
            adata.uns[f'mask_{b}'] = mask
        mdict[m] = adata

    if not mdict:
        raise ValueError(f"No modalities loaded from {p}; nothing in feat_dims.toml matched files on disk.")

    mdata = mu.MuData(mdict)
    mdata.uns['feat_dims'] = {m: list(map(int, v)) for m, v in feat_dims.items() if m in mdict}

    # Push 'batch' to top-level mdata.obs so plotting tools that read
    # mdata.obs can find it without modality prefixes. For each cell in mdata.obs_names,
    # take 'batch' from whichever modality contains it (they agree by
    # construction).
    batch_top = pd.Series(index=mdata.obs_names, dtype=object)
    for m, ad_m in mdict.items():
        for cid, b in zip(ad_m.obs_names, ad_m.obs['batch'].astype(str)):
            if pd.isna(batch_top.get(cid, np.nan)):
                batch_top.loc[cid] = b
    mdata.obs['batch'] = pd.Categorical(batch_top.values)

    if label_dir is not None:
        ld = Path(label_dir)
        for m in mdict:
            label_series = pd.Series(index=mdict[m].obs_names, dtype=object)
            for b, group_cells in mdict[m].obs.groupby('batch', observed=True):
                lf = ld / f'{b}.csv'
                if not lf.exists():
                    continue
                labels = pd.read_csv(lf, index_col=0).iloc[:, 0].astype(str).values
                if len(labels) != len(group_cells):
                    logger.warning(
                        "Label file %s has %d rows but modality %r has %d cells in batch %s; skipping labels for that batch.",
                        lf, len(labels), m, len(group_cells), b,
                    )
                    continue
                label_series.loc[group_cells.index] = labels
            mdict[m].obs[label_col] = pd.Categorical(label_series.values)

        # Push the label to top-level mdata.obs the same way as 'batch'.
        label_top = pd.Series(index=mdata.obs_names, dtype=object)
        for m, ad_m in mdict.items():
            if label_col not in ad_m.obs.columns:
                continue
            for cid, lab in zip(ad_m.obs_names, ad_m.obs[label_col].astype(str)):
                if pd.isna(label_top.get(cid, np.nan)):
                    label_top.loc[cid] = lab
        mdata.obs[label_col] = pd.Categorical(label_top.values)

    logger.info(
        "from_dir: loaded %s modalities (%s) across %d batches.",
        len(mdict), list(mdict.keys()), len(batch_dirs),
    )
    return mdata
