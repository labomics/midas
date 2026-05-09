Data layout (input and output)
==============================

This page describes the **data contract** between your data and MIDAS — what
shape MIDAS expects on input, and where it writes its results on output.

Recommended path: bring a :class:`MuData`. The legacy directory format is still
supported but is now considered an advanced / reproducibility option (see the
end of this page).

Input: a MuData
~~~~~~~~~~~~~~~

MIDAS accepts a single :class:`mudata.MuData` containing one
:class:`anndata.AnnData` per modality. Three things must be set:

.. list-table::
   :widths: 6 14
   :header-rows: 1

   * - Where
     - What
   * - ``mdata[m].X``
     - Per-modality counts (or whatever ``trsf_before_enc_<m>`` expects).
       MIDAS applies its own ``log1p`` / ``binarize`` internally, so for
       RNA / ADT / ATAC just store **raw counts** here.
   * - ``mdata[m].obs[batch_key]``
     - A column (default name ``'batch'``) identifying the source batch.
       Required even when there is only one batch — MIDAS uses it to know
       how cells partition across batches.
   * - ``mdata[m].uns[f'mask_{batch}']`` *(optional)*
     - A 1-D float array of length ``n_features``. ``1`` keeps a feature,
       ``0`` masks it out for that batch / modality combination. Use this
       for cross-batch feature alignment when not all batches share the
       same feature set. If absent, MIDAS treats every feature as present.

For ATAC encoded by chromosome chunk, also set:

.. code-block:: python

    mdata.uns['feat_dims'] = {'atac': [chunk1_size, chunk2_size, ...]}

and pass ``dims_x=mdata.uns['feat_dims']`` to :func:`MIDAS.setup_mudata`.

Quickstart
^^^^^^^^^^

The minimal "I have a MuData, run MIDAS" pipeline:

.. code-block:: python

    import scmidas

    scmidas.MIDAS.setup_mudata(mdata, batch_key='batch')
    model = scmidas.MIDAS(mdata)
    model.train(max_epochs=2000)

    mdata.obsm['X_midas']   = model.get_latent_representation()           # biological c
    mdata.obsm['X_midas_u'] = model.get_latent_representation(kind='u')   # technical u

If you do not yet have a MuData, see the
:doc:`Preparing your data <preparing_your_data>` tutorial for a full
scanpy-native pipeline starting from raw 10x output.

Output: written back to the MuData
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MIDAS writes its results to standard scanpy locations on the MuData, so any
downstream tool that reads ``mdata.obsm`` works out of the box.

.. list-table::
   :widths: 12 14
   :header-rows: 1

   * - Key
     - What
   * - ``mdata.obsm['X_midas']``
     - Biological joint latent ``z_c`` of shape ``(n_obs, dim_c)``,
       written by :func:`scmidas.integrate` or
       ``model.get_latent_representation(kind='c')``. Pass directly to
       ``sc.pp.neighbors(use_rep='X_midas')``.
   * - ``mdata.obsm['X_midas_u']``
     - Technical joint latent ``z_u`` of shape ``(n_obs, dim_u)``.
   * - imputed counts
     - Returned as an array by ``model.get_imputed_values(modality='rna')``,
       shape ``(n_obs, n_features)``. Assign wherever you prefer
       (e.g. ``mdata['rna'].layers['imputed']`` for the cells that were in
       that modality, or ``mdata.obsm['rna_imputed']`` for all cells).

For the full prediction surface (per-modality latents, batch-corrected
reconstructions, modality translation), see :meth:`scmidas.MIDAS.predict`.

Bridging from your AnnData to MuData
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your data is already in scanpy form, the bridge is a one-liner per
modality plus the :class:`MuData` constructor:

.. code-block:: python

    import mudata as mu

    # adata_rna, adata_adt: your already-QC'd, HVG-selected AnnDatas
    adata_rna.obs['batch'] = adata_rna.obs['donor']      # whatever your batch col is named
    adata_adt.obs['batch'] = adata_adt.obs['donor']

    mdata = mu.MuData({'rna': adata_rna, 'adt': adata_adt})

For the QC / normalization / HVG steps that lead up to this point, see the
:doc:`Preparing your data <preparing_your_data>` tutorial.

Bridging from a MIDAS directory dataset to a MuData
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository includes a helper that loads the legacy directory format
(``mat/<modality>.mtx``, ``mask/<modality>.csv``, ``feat/feat_dims.toml``)
into a :class:`MuData` directly:

.. code-block:: python

    mdata = scmidas.datasets.from_dir(
        'dataset/wnn_full_8batch_mtx/data',
        label_dir='dataset/wnn_full_8batch_mtx/label',  # optional cell labels
    )
    scmidas.MIDAS.setup_mudata(mdata, batch_key='batch',
                               dims_x=mdata.uns['feat_dims'])  # only if ATAC chunks
    model = scmidas.MIDAS(mdata)

Advanced: directory format
~~~~~~~~~~~~~~~~~~~~~~~~~~

The directory format below is the on-disk layout produced by the
preprocessing scripts on the
`reproducibility branch <https://github.com/labomics/midas/tree/reproducibility>`_,
and is what the bundled ``demo1`` / ``demo2`` / ``demo3`` datasets ship as.
For most users, :func:`scmidas.datasets.from_dir` (above) is the simplest
way to consume it; but if you want to read the directory format
directly, :meth:`MIDAS.configure_data_from_dir` accepts it:

.. code-block:: bash

    ./dataset_path/
        feat/
            feat_dims.toml
        batch_0/
            mat/<modality>.mtx
            mask/<modality>.csv  # optional, 1-row 0/1 CSV per modality
            cell_names.csv       # optional, one cell ID per row
        batch_1/
            ...

.. note::

   :meth:`MIDAS.configure_data_from_mdata` and
   :meth:`MIDAS.configure_data_from_dir` are kept for backwards
   compatibility but emit a ``DeprecationWarning``. Use
   :func:`MIDAS.setup_mudata` + :class:`MIDAS` for new code.
