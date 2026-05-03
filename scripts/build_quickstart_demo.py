"""Build the lightweight quickstart MuData from wnn_mosaic_8batch_mtx.

Output: a single .h5mu file roughly 3-5 MB containing a representative
mosaic subset (4 batches, ~600 cells each, 500 HVGs + all 224 ADT
features). Reproducible (seed=42).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread

warnings.filterwarnings('ignore')
sc.settings.verbosity = 0

SEED = 42
N_CELLS_PER_BATCH = 400
N_HVG = 500

# Picked to preserve the full 1+1+2 mosaic structure of the source dataset
# while keeping the result < 10MB. p3_0 brings NK over-representation;
# p4_0 brings the more typical CD4/Mono/CD8 mix.
SELECTED_BATCHES: list[str] = ['p1_0', 'p2_0', 'p3_0', 'p4_0']
MODALITIES_PER_BATCH: dict[str, list[str]] = {
    'p1_0': ['rna'],
    'p2_0': ['adt'],
    'p3_0': ['rna', 'adt'],
    'p4_0': ['rna', 'adt'],
}


def stratified_subsample(celltypes: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'idx': np.arange(len(celltypes)), 'ct': celltypes})
    n = len(celltypes)
    if n_target >= n:
        return df['idx'].values
    chosen = []
    for ct, group in df.groupby('ct'):
        k = max(1, int(round(n_target * len(group) / n)))
        k = min(k, len(group))
        chosen.extend(rng.choice(group['idx'].values, size=k, replace=False).tolist())
    chosen = np.array(chosen)
    if len(chosen) > n_target:
        chosen = rng.choice(chosen, size=n_target, replace=False)
    elif len(chosen) < n_target:
        leftover = np.setdiff1d(df['idx'].values, chosen)
        extra = rng.choice(leftover, size=n_target - len(chosen), replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen)


def select_hvgs(src: Path, rna_names: list[str], rna_batches: list[str]) -> list[str]:
    """Compute Seurat batch-aware HVGs across the *selected* RNA-having batches."""
    adatas = []
    for b in rna_batches:
        m = mmread(src / f'data/{b}/mat/rna.mtx').tocsr().astype(np.float32)
        a = ad.AnnData(X=m)
        a.var_names = rna_names
        a.obs_names = [f'{b}_{i}' for i in range(m.shape[0])]
        a.obs['batch'] = b
        adatas.append(a)
    combined = ad.concat(adatas, join='outer')
    combined.obs_names_make_unique()
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    sc.pp.highly_variable_genes(combined, n_top_genes=N_HVG, batch_key='batch', flavor='seurat')
    return combined.var.index[combined.var['highly_variable']].tolist()


def build(src_dir: str, out_path: str) -> mu.MuData:
    src = Path(src_dir)

    rna_names = pd.read_csv(src / 'data/feat/feat_names_rna.csv', index_col=0)['x'].tolist()
    adt_names = pd.read_csv(src / 'data/feat/feat_names_adt.csv', index_col=0)['x'].tolist()
    print(f'Source: {len(rna_names)} RNA features, {len(adt_names)} ADT features')

    rna_batches = [b for b in SELECTED_BATCHES if 'rna' in MODALITIES_PER_BATCH[b]]
    adt_batches = [b for b in SELECTED_BATCHES if 'adt' in MODALITIES_PER_BATCH[b]]
    print(f'Selected batches: {SELECTED_BATCHES}')
    print(f'  RNA batches: {rna_batches}')
    print(f'  ADT batches: {adt_batches}')

    print(f'\nComputing {N_HVG} HVGs across selected RNA batches...')
    hvgs = select_hvgs(src, rna_names, rna_batches)
    print(f'  selected {len(hvgs)} HVGs')

    print(f'\nStratified subsample ({N_CELLS_PER_BATCH} cells/batch)...')
    selected_idx: dict[str, np.ndarray] = {}
    for b in SELECTED_BATCHES:
        lbl = pd.read_csv(src / f'label/{b}.csv', index_col=0)
        celltypes = lbl.iloc[:, 0].values
        idx = stratified_subsample(celltypes, N_CELLS_PER_BATCH, SEED)
        selected_idx[b] = idx
        ct_dist = pd.Series(celltypes[idx]).value_counts().head(4).to_dict()
        print(f'  {b}: {len(idx)} cells | top: {ct_dist}')

    hvg_mask = np.isin(rna_names, hvgs)

    rna_adatas = []
    adt_adatas = []
    for b in SELECTED_BATCHES:
        idx = selected_idx[b]
        cell_ids = [f'{b}_{i}' for i in idx]
        lbl = pd.read_csv(src / f'label/{b}.csv', index_col=0)
        celltypes = lbl.iloc[:, 0].values[idx]

        if 'rna' in MODALITIES_PER_BATCH[b]:
            m = mmread(src / f'data/{b}/mat/rna.mtx').tocsr().astype(np.float32)
            m = m[idx, :][:, hvg_mask]
            a = ad.AnnData(X=m)
            a.var_names = [n for n, k in zip(rna_names, hvg_mask) if k]
            a.obs_names = cell_ids
            a.obs['batch'] = b
            a.obs['celltype'] = celltypes
            rna_adatas.append(a)

        if 'adt' in MODALITIES_PER_BATCH[b]:
            m = mmread(src / f'data/{b}/mat/adt.mtx').tocsr().astype(np.float32)
            m = m[idx, :]
            a = ad.AnnData(X=m)
            a.var_names = adt_names
            a.obs_names = cell_ids
            a.obs['batch'] = b
            a.obs['celltype'] = celltypes
            adt_adatas.append(a)

    rna_full = ad.concat(rna_adatas, join='outer')
    adt_full = ad.concat(adt_adatas, join='outer')
    print(f'\nAnnData shapes: rna={rna_full.shape}  adt={adt_full.shape}')

    mdata = mu.MuData({'rna': rna_full, 'adt': adt_full})
    mdata.update()

    # Promote batch / celltype from modality obs to top-level mdata.obs so
    # users only need to look in one place.
    batch_per_cell: dict[str, str] = {}
    ct_per_cell: dict[str, str] = {}
    for mod_adata in [rna_full, adt_full]:
        for cid, b, ct in zip(mod_adata.obs_names, mod_adata.obs['batch'], mod_adata.obs['celltype']):
            batch_per_cell.setdefault(cid, b)
            ct_per_cell.setdefault(cid, ct)
    mdata.obs['batch'] = pd.Series(batch_per_cell)
    mdata.obs['celltype'] = pd.Series(ct_per_cell)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mdata.write_h5mu(str(out))
    size_mb = out.stat().st_size / 1024 / 1024
    print(f'\nWrote {out}: {size_mb:.2f} MB')

    return mdata


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='/tmp/midas-demo2/wnn_mosaic_8batch_mtx',
                   help='Path to extracted wnn_mosaic_8batch_mtx directory')
    p.add_argument('--out', default='/tmp/quickstart_pbmc_mosaic.h5mu',
                   help='Output .h5mu path')
    args = p.parse_args()
    build(args.src, args.out)
