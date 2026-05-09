"""Plotting helpers for MIDAS / MuData users.

The high-level helpers (:func:`umap`, :func:`modality_grid`) take a
:class:`MuData` directly and route through a temporary :class:`AnnData`
wrapper, side-stepping the current limitations of scanpy + MuData
plotting. They are exposed as both ``scmidas.plot.X`` and the shorter
``scmidas.pl.X``.

The AnnData-only helpers :func:`plot_umap`, :func:`plot_umap_grid`,
:func:`plot_z_umap_grid` are retained for backwards compatibility with
older tutorial code.
"""
from __future__ import annotations

import copy
import logging
from typing import Iterable, Optional, Sequence, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


def umap(
    mdata,
    *,
    basis: str = 'X_midas',
    color: Union[str, Sequence[str]] = 'batch',
    obs_keys: Optional[Sequence[str]] = None,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    random_state: int = 42,
    shuffle: bool = True,
    recompute: bool = False,
    **kwargs,
):
    """Compute and plot a UMAP from a MuData embedding (one-liner).

    Wraps the MuData's selected embedding as a thin AnnData and routes
    through ``sc.pl.umap``, avoiding the current limitations of scanpy
    + MuData plotting.

    Parameters:
        mdata : MuData
            Multi-modal data with an integration embedding in
            ``mdata.obsm[basis]`` (e.g. written by
            :meth:`~scmidas.model.MIDAS.get_latent_representation`).
        basis : str
            Key in ``mdata.obsm`` to use as the representation. Default
            ``'X_midas'`` matches :meth:`MIDAS.get_latent_representation`.
        color : str or sequence of str
            One or more ``mdata.obs`` columns to color by. Mirrors
            scanpy's ``color`` argument.
        obs_keys : sequence of str, optional
            Which ``mdata.obs`` columns to copy onto the temporary
            AnnData. Defaults to the union of ``color`` plus any keys
            referenced by scanpy kwargs that read from ``.obs``.
        n_neighbors, min_dist, random_state
            Forwarded to ``sc.pp.neighbors`` / ``sc.tl.umap``.
        shuffle : bool
            If True, shuffle cells before plotting (unbiased visual
            density when batches/cell types overlap).
        recompute : bool
            If False (default) and ``mdata.obsm['X_umap_<basis>']``
            already exists, reuse it. If True, recompute UMAP.
        **kwargs
            Forwarded to ``sc.pl.umap``.

    Returns:
        AnnData: The temporary AnnData used for plotting (so callers can
        access ``.obsm['X_umap']`` if they want to keep it).
    """
    if basis not in mdata.obsm:
        raise KeyError(
            f"mdata.obsm[{basis!r}] not found. Run "
            f"model.get_latent_representation() and assign it first."
        )
    color_list = [color] if isinstance(color, str) else list(color)
    keep_cols = list(set((obs_keys or []) + color_list))
    keep_cols = [c for c in keep_cols if c in mdata.obs.columns]
    obs = mdata.obs[keep_cols].copy() if keep_cols else None

    cache_key = f'X_umap_{basis}'
    have_cache = cache_key in mdata.obsm and not recompute

    adata = ad.AnnData(X=mdata.obsm[basis], obs=obs)
    adata.obs_names = mdata.obs_names

    if have_cache:
        adata.obsm['X_umap'] = mdata.obsm[cache_key]
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
        sc.tl.umap(adata, random_state=random_state, min_dist=min_dist)
        mdata.obsm[cache_key] = adata.obsm['X_umap']

    if shuffle:
        sc.pp.subsample(adata, fraction=1, random_state=random_state)

    sc.pl.umap(adata, color=color_list, **kwargs)
    return adata


def modality_grid(
    model,
    mdata,
    *,
    batch_key: str = 'batch',
    label_key: str = 'label',
    figsize: float = 2.0,
    point_size: float = 2.0,
    fontsize: int = 10,
    transpose: bool = False,
    random_state: int = 42,
):
    """Per-modality vs per-batch UMAP grid (joint c plus single-modality c).

    Internally runs ``model.predict(joint_latent=True, mod_latent=True)``,
    concatenates the per-modality biological latents, and tiles them as
    a (modality × batch) grid coloured by ``label_key``. The grid view
    answers: "does each modality on its own carry enough signal to
    separate cell types in each batch?"

    Parameters:
        model : MIDAS
            A constructed (and trained or checkpoint-loaded) MIDAS model.
        mdata : MuData
            The MuData passed to ``MIDAS(mdata)``.
        batch_key : str
            Column in ``mdata.obs`` (or ``mdata[m].obs``) identifying
            the batch.
        label_key : str
            Column in ``mdata.obs`` (or ``mdata[m].obs``) used for
            colouring.
        figsize, point_size, fontsize : float
            Per-subplot styling.
        transpose : bool
            If True, swap rows/columns (modality-by-batch instead of
            batch-by-modality).
        random_state : int
            Seed used by ``sc.tl.umap``.

    Returns:
        AnnData: The aggregated AnnData used for the plot.
    """
    out = model.predict(joint_latent=True, mod_latent=True, verbose=False)

    pieces, types_, batches_, labels_ = [], [], [], []
    for batch_id, b in enumerate(model.batch_names):
        block = out[b]
        first_mod = model.combs[batch_id][0]
        # cell IDs aligned to the FIRST modality in this batch
        sub_obs = mdata[first_mod].obs[mdata[first_mod].obs[batch_key].astype(str) == str(b)]
        sub_labels = sub_obs[label_key].astype(str).values if label_key in sub_obs.columns else np.full(len(sub_obs), '?')
        for k, z in block.get('z_c', {}).items():
            if z.shape[0] != len(sub_obs):
                continue
            pieces.append(z)
            types_.append(np.full(z.shape[0], k.upper()))
            batches_.append(np.full(z.shape[0], b))
            labels_.append(sub_labels)

    if not pieces:
        raise RuntimeError(
            "No per-modality latents to plot. Was the model trained?"
        )

    X = np.concatenate(pieces)
    adata = ad.AnnData(X=X)
    adata.obs['type'] = pd.Categorical(np.concatenate(types_))
    adata.obs[batch_key] = pd.Categorical(np.concatenate(batches_))
    adata.obs[label_key] = pd.Categorical(np.concatenate(labels_))

    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata, random_state=random_state)

    axis1 = batch_key if not transpose else 'type'
    axis2 = 'type' if not transpose else batch_key
    rows = adata.obs[axis1].cat.categories.tolist()
    cols = adata.obs[axis2].cat.categories.tolist()
    # Preferred display order for the modality axis: ATAC, RNA, ADT, JOINT
    # (any modalities outside this list keep their original ordering and
    # come last).
    _preferred = ['ATAC', 'RNA', 'ADT', 'JOINT']
    def _reorder(types):
        front = [t for t in _preferred if t in types]
        rest = [t for t in types if t not in _preferred]
        return front + rest
    if transpose:
        rows = _reorder(rows)
    else:
        cols = _reorder(cols)
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(figsize * len(cols), figsize * len(rows)))
    axes = np.atleast_2d(axes)

    fig_dummy, ax_dummy = plt.subplots()
    sc.pl.umap(adata, color=label_key, show=False, ax=ax_dummy)
    handles, leg_labels = ax_dummy.get_legend_handles_labels()
    plt.close(fig_dummy)

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            sc.pl.umap(adata, show=False, ax=ax, s=point_size)  # background
            sub = adata[(adata.obs[axis1] == r) & (adata.obs[axis2] == c)]
            if sub.n_obs > 0:
                sc.pl.umap(sub, color=label_key, show=False, ax=ax, s=point_size)
                if ax.get_legend():
                    ax.get_legend().set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(''); ax.set_title(c if i == 0 else '')
            ax.set_ylabel(r if j == 0 else '')

    fig.legend(handles, leg_labels, loc='center', bbox_to_anchor=(0.5, -0.02),
               ncol=len(leg_labels), fontsize=fontsize)
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.show()
    return adata


# ---------------------------------------------------------------------------
# Legacy AnnData-only helpers (used by the original tutorials)
# ---------------------------------------------------------------------------
def plot_umap(
    adata, 
    key='z_c_joint', 
    do_pca=False, 
    n_comps=32, 
    color='batch', 
    shuffle=True,
    **kwargs
    ):

    """
    Computes and plots a UMAP for a single AnnData object based on a specific latent representation.

    This function allows for optional PCA preprocessing on the selected latent representation 
    (stored in .obsm) before computing the neighborhood graph and UMAP.

    Args:
        adata (AnnData): The input annotated data matrix.
        key (str, optional): The key in `adata.obsm` to use as the representation. 
                             Defaults to 'z_c_joint'.
        do_pca (bool, optional): Whether to perform scaling and PCA on the representation 
                                 before neighbor calculation. Defaults to False.
        n_comps (int, optional): The number of principal components to use if `do_pca` is True. 
                                 Defaults to 32.
        color (str, optional): The key in `adata.obs` used to color the plot. 
                               Defaults to 'batch'.
        shuffle (bool, optional): Shuffle the samples.
        **kwargs: Additional keyword arguments passed to `sc.pl.umap`.

    Returns:
        None: Displays the plot.
    """

    if do_pca:
        adata2 = sc.AnnData(adata.obsm[key])
        adata2.obs = adata.obs
        sc.pp.scale(adata2)
        sc.pp.pca(adata2, n_comps=n_comps)
        key = 'X_pca'
    else:
        adata2 = copy.deepcopy(adata)
    if shuffle:
        sc.pp.subsample(adata2, fraction=1)
    sc.pp.neighbors(adata2, use_rep=key)
    sc.tl.umap(adata2)
    sc.pl.umap(adata2, color=color, **kwargs)

def plot_umap_grid(adata, axis1, axis2, color, figsize=2, point_size=2, fontsize=10, background=True):
    """
    Plots a grid (facet plot) of UMAPs split by two categorical variables.

    This visualizes how specific groups (defined by axis1 and axis2) are distributed 
    within the global UMAP space.

    Args:
        adata (AnnData): Annotated data matrix with pre-computed UMAP coordinates (`X_umap`).
        axis1 (str): Key in `adata.obs` defining the rows of the grid.
        axis2 (str): Key in `adata.obs` defining the columns of the grid.
        color (str): Key in `adata.obs` used for coloring the points.
        figsize (float, optional): The size (in inches) of each subplot. Defaults to 2.
        point_size (float, optional): The size of the scatter points. Defaults to 2.
        fontsize (int, optional): Font size for the legend. Defaults to 10.
        background (bool, optional): If True, plots all cells in grey in the background 
                                     of each subplot to show the global structure. Defaults to True.

    Returns:
        None: Displays the plot.
    """
    axis1_names = adata.obs[axis1].unique() 
    axis2_names = adata.obs[axis2].unique()
    nrows = len(axis1_names)
    ncols = len(axis2_names)
    fig, ax = plt.subplots(nrows, ncols, figsize=[figsize * ncols, figsize * nrows])
    fig_dummy, ax_dummy = plt.subplots()
    sc.pl.umap(adata, color=color, show=False, ax=ax_dummy)
    handles, labels_ = ax_dummy.get_legend_handles_labels()
    plt.close(fig_dummy)
    for i, k1 in enumerate(axis1_names):
        for j, k2 in enumerate(axis2_names):
            if background:
                sc.pl.umap(adata, show=False, ax=ax[i, j], s=point_size) # background
            sc.pl.umap(adata[(adata.obs[axis1]==k1) & (adata.obs[axis2]==k2)], color=color, show=False, ax=ax[i, j], s=point_size)
            ax[i, j].get_legend().set_visible(False)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_xlabel('')
            if j==0:
                ax[i, j].set_ylabel(k1)
            else:
                ax[i, j].set_ylabel('')
            if i==0:
                ax[i, j].set_title(k2)
            else:
                ax[i, j].set_title('')
    # create global legend
    fig.legend(handles, labels_, loc='center', 
               bbox_to_anchor=(0.5, -0.02), ncol=len(labels_), fontsize=fontsize)
    # adjust the figure
    plt.tight_layout(rect=[0.1, 0.05, 1, 1])
    plt.show()

def plot_z_umap_grid(adata_list, batch_col='batch', color='label', figsize=2, point_size=2, fontsize=10, transpose=False):
    """
    Aggregates latent representations from a dictionary of AnnData objects, computes a joint UMAP, 
    and plots a grid view.

    It specifically looks for keys in `.obsm` starting with 'z_c', concatenates them, 
    and re-computes the UMAP to visualize the alignment or distribution across different batches/types.

    Args:
        adata_list (dict): A dictionary where keys are batch identifiers and values are AnnData objects.
        batch_col (str, optional): Key in `adata.obs` identifying the batch/sample. Defaults to 'batch'.
        color (str, optional): Key in `adata.obs` used for coloring. Defaults to 'label'.
        figsize (float, optional): The size (in inches) of each subplot. Defaults to 2.
        point_size (float, optional): The size of the scatter points. Defaults to 2.
        fontsize (int, optional): Font size for the legend. Defaults to 10.
        transpose (bool, optional): If True, swaps the row and column axes of the grid 
                                    (Batch vs. Type). Defaults to False.

    Returns:
        None: Displays the plot.
    """
    data = []
    axis1_ = []
    axis2_ = []
    label_ = []
    for b, adata in adata_list.items():
        for k in adata.obsm:
            if k.startswith('z_c'):
                data.append(adata.obsm[k])
                axis1_.append(adata.obs[batch_col])
                axis2_.append([k.split('_')[-1].upper() for i in range(len(adata))])
                label_.append(adata.obs[color])

    data =  np.concatenate(data)
    axis1_ = np.concatenate(axis1_)
    axis2_ = np.concatenate(axis2_)
    label_ = np.concatenate(label_)
    adata = sc.AnnData(data)
    adata.obs['batch'] = axis1_
    adata.obs['type'] = axis2_
    adata.obs[color] = label_

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    axis1 = 'batch' if not transpose else 'type'
    axis2 = 'type' if not transpose else 'batch'
    axis1_names = adata.obs[axis1].unique() 
    axis2_names = adata.obs[axis2].unique()
    nrows = len(axis1_names)
    ncols = len(axis2_names)
    fig, ax = plt.subplots(nrows, ncols, figsize=[figsize * ncols, figsize * nrows])
    fig_dummy, ax_dummy = plt.subplots()
    sc.pl.umap(adata, color=color, show=False, ax=ax_dummy)
    handles, labels_ = ax_dummy.get_legend_handles_labels()
    plt.close(fig_dummy)
    for i, k1 in enumerate(axis1_names):
        for j, k2 in enumerate(axis2_names):
            sc.pl.umap(adata, show=False, ax=ax[i, j], s=point_size) # background
            sc.pl.umap(adata[(adata.obs[axis1]==k1) & (adata.obs[axis2]==k2)], color=color, show=False, ax=ax[i, j], s=point_size)
            ax[i, j].get_legend().set_visible(False)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_xlabel('')
            if j==0:
                ax[i, j].set_ylabel(k1)
            else:
                ax[i, j].set_ylabel('')
            if i==0:
                ax[i, j].set_title(k2)
            else:
                ax[i, j].set_title('')
    # create global legend
    fig.legend(handles, labels_, loc='center', 
               bbox_to_anchor=(0.5, -0.02), ncol=len(labels_), fontsize=fontsize)
    # adjust the figure
    plt.tight_layout(rect=[0.1, 0.05, 1, 1])
    plt.show()