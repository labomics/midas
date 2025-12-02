import scanpy as sc 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
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