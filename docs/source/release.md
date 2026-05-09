# Release Notes

All notable changes to this project will be documented in this file.

---

## Version 0.3.x

### v0.3.0 (2026-05-09)

Major refresh of the user-facing API around a single :class:`MuData`. The new entry points (`setup_mudata`, `MIDAS(mdata)`, `get_latent_representation`, `get_imputed_values`, `save` / `load`) compose directly with `mdata.obsm`, `sc.pp.neighbors(use_rep=...)`, and the rest of the standard single-cell stack. A new plotting namespace `scmidas.pl` and a data-prep tutorial round out the package for users coming straight from raw 10x output.

*   **🚀 New — `MIDAS` entry points centred on `MuData`**
    *   `MIDAS.setup_mudata(mdata, batch_key=...)` — register a MuData (writes config to `mdata.uns['_scmidas']`).
    *   `MIDAS(mdata, ...)` — construct directly from a registered MuData; instance state instead of class-level state (fixes a multi-instance interference bug).
    *   `model.get_latent_representation(kind='c'|'u'|'joint')` — returns the joint latent aligned to `mdata.obs_names`. Drop straight into `mdata.obsm['X_midas']`.
    *   `model.get_imputed_values(modality='rna')` — returns imputed counts aligned to `mdata.obs_names`.
    *   `model.save(dir)` / `MIDAS.load(dir, mdata)` — symmetric save/load (writes `model.pt` + `setup.json`).
    *   `MIDAS(mdata)` now defaults to `transform={'atac': 'binarize'}` whenever `'atac'` is among the modalities (override by passing your own `transform` dict).
*   **🚀 New — `scmidas.pl` plotting namespace**
    *   `scmidas.pl.umap(mdata, basis='X_midas', color=[...])` — one-line UMAP that works around the current scanpy + MuData plotting limitations via a thin AnnData wrapper.
    *   `scmidas.pl.modality_grid(model, mdata, label_key=...)` — collapses the per-modality vs per-batch grid (~22 lines in the previous demos) into one call. Modality columns are ordered ATAC, RNA, ADT, Joint when present.
*   **🚀 New — `scmidas.datasets.from_dir`**
    *   Loads the directory-format datasets (`mat/<m>.mtx`, `mask/<m>.csv`, `feat/feat_dims.toml`) into a `MuData`, including masks, labels, and ATAC chunk dims.
*   **📚 New tutorial — Preparing your data**
    *   `docs/source/tutorials/basics/preparing_your_data.ipynb` walks from a public 10x Genomics 5k PBMC CITE-seq sample through QC, HVG selection, MuData wrap, MIDAS integration, Leiden clustering, and a synthetic mosaic example.
*   **📚 Docs cleanup**
    *   `inputs.rst` + `outputs.rst` merged into `data_layout.rst` — a single page describing the MuData input/output contract. The directory format is moved to an "advanced" section.
    *   All three demos (`demo1`, `demo2`, `demo3`) rewritten to use the new API: `from_dir` → `setup_mudata` → `MIDAS(mdata)` → `get_latent_representation`. The 22-line per-modality grid block became `scmidas.pl.modality_grid(model, mdata)`. Each demo gained a 6.4 "After integration" section (Leiden + UMAP).
    *   README adds a "Bring your own data" section linking the new tutorial and the data-layout reference.
*   **🛠 Backwards compatibility**
    *   `MIDAS.configure_data_from_mdata` and `MIDAS.configure_data_from_dir` still work — they emit a `DeprecationWarning` and will be removed in 0.4.0.
    *   `save_checkpoint` / `load_checkpoint` still work; new code should use `save` / `load`.
*   **🐛 Fixes**
    *   `predict(joint_latent=False)` no longer raises `KeyError: 'z_c'`.
    *   Multiple `MIDAS()` instances in one process now have independent state (was previously class-level — a second instance would clobber the first).

---

## Version 0.2.x

### v0.2.0 (2026-05-03)
*   **🚀 New — `scmidas.integrate(mdata)` one-line entry point**
    *   A thin top-level wrapper around `MIDAS.configure_data_from_mdata`
        + `train()` with toy-tuned defaults (`batch_size=128`,
        `max_epochs=65`, `lr=3e-4`) so that the bundled quickstart
        dataset converges in roughly one minute on a single mid-range
        GPU. The full `MIDAS` class API is unchanged for users who
        need control.
    *   ⚠️ The defaults are tuned for the toy quickstart only. For
        real datasets, override `max_epochs` (1000-2000) and consider
        `batch_size=256`.
*   **🚀 New — bundled quickstart dataset**
    *   `scmidas.datasets.quickstart()` returns a 1600-cell PBMC RNA+ADT
        mosaic MuData (4 batches, full mosaic structure: one RNA-only,
        one ADT-only, two paired). 500 RNA HVGs + 224 ADT features,
        2.66 MB shipped inside the wheel.
    *   Source: hand-tuned subset of `wnn_mosaic_8batch_mtx`. Build
        script: `scripts/build_quickstart_demo.py`.
*   **📚 Documentation**
    *   New `examples/quickstart.ipynb` — pre-rendered notebook that
        users can open in Colab via the new badge in the README, no
        local install required.
    *   README quickstart rewritten: replaces the previous `...` API
        sketch with a runnable five-line snippet using
        `scmidas.datasets.quickstart()` + `scmidas.integrate()`,
        followed by the rendered UMAP image.
*   **⚙️ Packaging**
    *   `pyproject.toml` ships `data/*.h5mu` as package data so the
        quickstart dataset travels with the wheel.
    *   Module-level `logging.basicConfig(level=INFO)` removed from
        five files (`config`, `data`, `model`, `nn`, `utils`); each
        now does the canonical `logger = logging.getLogger(__name__)`
        instead. Demo notebooks call `logging.basicConfig` themselves
        so visible output is unchanged. Libraries should not call
        `basicConfig` — it overrides the user's own logging config.

## Version 0.1.x

### v0.1.19 (2026-05-03)
*   **📦 Packaging — narrow torch upper bound to `<2.11`**
    *   torch 2.11 dropped Volta (V100, CC 7.0) and Pascal (P100, GTX
        10xx, CC 6.x) from its default `cu128` / `cu129` wheels (to
        ship cuDNN 9.15.1, which is incompatible with those archs). On
        those GPUs `pip install scmidas==0.1.18` would silently install
        a torch that fails at the first CUDA op with
        `no kernel image is available for execution on the device`.
    *   The pin now reads `torch>=2.5,<2.11` (with matching
        `torchvision<0.26` / `torchaudio<2.11`). Users on
        Ampere/Hopper/Ada/Blackwell GPUs can manually upgrade past the
        cap; users on Volta/Pascal stay on a working default install.
    *   No source-code change — same scmidas as 0.1.18.
*   **✨ Enhancements**
    *   `import scmidas` now runs a one-time GPU self-check: if the
        local torch wheel has no kernels for the local GPU, scmidas
        emits a `UserWarning` with actionable guidance (downgrade torch
        or use the cu126 wheel) instead of the user later seeing a raw
        `no kernel image is available` error from somewhere deep in
        their training loop. The check no-ops on CPU-only environments
        and on working GPU setups.
*   **⚙️ CI**
    *   Test matrix gained a `torch 2.10` job (the new upper bound) and
        dropped the previous experimental `torch latest` job. Lower
        bound remains `torch 2.5.1` across Python 3.10 / 3.11 / 3.12.

### v0.1.18 (2026-05-02)
*   **🐛 Bug Fixes (DDP + mosaic data)**
    *   Default `sampler_type='auto'` now picks the DDP sampler when a
        process group is initialized. Previously `'auto'` silently fell
        back to `MultiBatchSampler` (a rank-agnostic sampler), so DDP
        runs computed each batch on every rank in parallel — correct
        but with no throughput gain over single-GPU. Users who already
        passed `sampler_type='ddp'` explicitly are unaffected.
    *   `MyDistributedSampler` now derives its shuffle order from a
        seeded `random.Random` instance (cross-rank-consistent for the
        dataset visit order, rank-specific for the within-dataset
        shuffle), and properly initialises the base
        `DistributedSampler`. Previously it used the global Python
        `random` module, so each DDP rank sampled a different sub-batch
        at the same step. With non-uniform per-sub-batch modality
        combinations (mosaic data), this produced different encoder
        graphs per rank and caused NCCL all-reduce to hang under
        `find_unused_parameters=False` (Lightning default), eventually
        triggering a watchdog timeout.
    *   **Heads-up — DDP reproducibility**: the DDP sampling order has
        changed as a side-effect of the fix. Existing seeded DDP runs
        will produce different numerics; checkpoints from prior
        versions still load and continue training, but the post-fix
        sampling sequence is not bit-equivalent to the pre-fix one.
        Single-GPU users (using `MultiBatchSampler`) are unaffected.
*   **🐛 Bug Fixes (API hardening)**
    *   `MIDAS.configure_optimizers` no longer raises `AttributeError`
        when entered through the simpler `configure_data` path
        (`load_optimizer_state` was only set by
        `configure_data_from_dir` / `configure_data_from_mdata` /
        `load_checkpoint`).
    *   `MIDAS.configure_data` default `batch_names` now use f-string
        formatting (`f'batch_{i}'`) instead of the literal string
        `'batch_%d'` repeated `len(datalist)` times.
    *   Bad ATAC configuration in `configure_data` now raises
        `ValueError` instead of calling `exit()` (which killed the
        Jupyter kernel without a traceback).
    *   `download_file` now accepts both `str` and `pathlib.Path` for
        `dest_path`. The signature was annotated `str` but the body
        called `.name`.
    *   `Encoder.forward` no longer mutates the caller's batch dict.
        The mask multiply is now out-of-place; the previous in-place
        `data[m] *= mask` corrupted upstream tensors for any modality
        without a `trsf_before_enc_*` transform. Mathematically
        equivalent (the mask is a 0/1 modality-presence indicator, and
        `calc_recon_loss` already multiplies the loss by the same
        mask), but makes the encoder safe to re-call on the same
        batch (e.g. `predict`'s `mod_latent` / `translate` paths).
    *   `VAE.forward` no longer wraps the PoE call in a bare
        `try/except` that swallowed real errors with a malformed
        `logging.debug` call.
*   **✅ Tests**
    *   Added `tests/test_invariants.py` pinning down the bugs above
        plus the DDP sampler determinism fix (cross-rank disjoint
        indices, `set_epoch` actually changes ordering).
*   **📚 Documentation**
    *   Each basics demo now exposes a single `# === GPU configuration ===`
        block (`GPUS` + `STRATEGY`) at the top so switching from
        single-GPU to multi-GPU only requires editing two values.
    *   Removed the redundant standalone `advanced/multi_gpu.rst`
        tutorial — its contents now live inline in the basics demos
        where the failure modes would actually be encountered.
    *   README: removed the duplicated MuData section (the `from_mdata`
        path is one link away in the docs), corrected the Quick
        Example comment about input format, and fixed the License
        badge link.
*   **⚙️ Packaging**
    *   Version is now single-sourced from `pyproject.toml`;
        `scmidas.__version__` and the Sphinx `release` both read it via
        `importlib.metadata.version("scmidas")` instead of duplicating
        the literal in three files.
    *   Relax the `torch` pin from `>=2.5,<2.6` to `>=2.5,<3` (and the
        matching `torchvision` / `torchaudio` companions). The previous
        `<2.6` cap was a workaround for a suspected Lightning-DDP
        incompatibility; torch 2.8 has now been verified end-to-end in
        the mosaic DDP path (1000-epoch run with UMAP and numerics
        consistent with the single-GPU baseline), so users on torch 2.6
        / 2.7 / 2.8 no longer have to manually override the pin.

### v0.1.17 (2026-03-17)
*   **🐛 Bug Fixes**
    *   Remove multi-threading for UMAP visualization during training.

### v0.1.16 (2026-03-05)
*   **✨ Enhancements**
    *   Asynchronous UMAP visualization during training
        *   UMAP plots can now be generated asynchronously to avoid blocking the training loop.
    *   Improved prediction API
        *   Flexible prediction outputs: choose between returning results in memory or saving directly to disk.
        *   Support streaming prediction to disk, enabling inference on large datasets with minimal memory usage.
        *   Added support for .npy format for faster saving and loading of prediction outputs.
    *   More flexible load_predicted function
        *   Allow loading specific batch names and variable groups (e.g. z_c, z_u, x_impt).
        *   Improves efficiency when working with large prediction outputs.
    *   Add model.train()
*   **🐛 Bug Fixes**
    *   Allow loading specific batch names and variable groups (e.g. z_c, z_u, x_impt).
    *   Delete init in the sampler class. Issue #28. 

### v0.1.15 (2025-11-28)
*   **✨ Enhancements**
    *   Update `MIDAS.predict()` functionality: 
        1. Added support for `AnnData` output format. 
        2. Optimized output handling to reduce GPU memory usage (offloaded outputs to CPU).
   *   Batch Handling: Added support for automatically fetching batch names.
*   **🐛 Bug Fixes**
    *   Removed redundant RNA and ADT layers.

### v0.1.13 (2025-08-28)
*   **📚 Documentation**
    *   Enhanced and clarified tutorials for a better user learning experience.

### v0.1.12 (2025-07-03)
*   **✨ Enhancements**
    *   Updated the source for fetching pre-trained models to ensure reliability.

### v0.1.10 (2025-07-02)
*   **✨ Enhancements**
    *   Adjusted default layer dimensions for the ATAC encoder/decoder (`dims_before_enc_atac=[128, 32]` and `dims_after_dec_atac=[32, 128]`) to improve model performance.

### v0.1.9 (2025-06-23)
*   **⚙️ Miscellaneous**
    *   Updated the minimum required Python version to `>=3.10`.

### v0.1.8 (2025-06-12)
*   **🚀 New Features**
    *   Added support for the `.mtx` (Matrix Market) input format for broader data compatibility.
    *   Introduced live UMAP visualization during training via TensorBoard. This can be enabled with `MIDAS.configure_data_from_dir(viz_umap_tb=True)`.
    *   Added a new utility function `data.download_models()` to easily fetch pre-trained models.
*   **✨ Enhancements**
    *   The `MIDAS.predict()` method now returns prediction results directly, improving efficiency and making it easier to chain operations.
    *   Updated the demonstration dataset with more relevant examples.
*   **🐛 Bug Fixes**
    *   Fixed a critical bug that prevented the optimizer from being re-initialized after loading a checkpoint with `MIDAS.load_checkpoint()`.
*   **📚 Documentation**
    *   Updated and expanded documentation and tutorials to reflect recent changes.

### v0.1.7 (2025-01-22)
*   **🐛 Bug Fixes**
    *   Resolved a bug reported in Issue #22.

### v0.1.6 (2025-01-20)
*   **🐛 Bug Fixes**
    *   Fixed a bug where the `dims_brefore_enc_atac` configuration was applied incorrectly. It is now conditionally used only when multiple ATAC input dimensions are provided.

### v0.1.5 (2025-01-17)
*   **🐛 Bug Fixes**
    *   Fixed an issue where Gaussian sampling was incorrectly performed during inference for modality-specific embeddings, leading to more deterministic outputs.

### v0.1.4 (2024-12-31)
*   **🐛 Bug Fixes**
    *   Corrected the data loading logic in `MIDAS.get_emb_umap()` by fixing the `load_predicted()` utility.

### v0.1.3 (2024-12-21)
*   **🚀 New Features**
    *   Integrated with **PyTorch Lightning** to enable streamlined multi-GPU training.
    *   Integrated with **TensorBoard** to facilitate real-time visualization of training and validation losses.
    *   Refactored the `MIDAS` architecture to support easier integration of new custom modalities.

---

## Version 0.0.x

### v0.0.18 (2024-07-29)
*   **🐛 Bug Fixes**
    *   In `utils.viz_mod_latent()`, rotated the visualization for better interpretation and fixed a bug that caused an error when processing a batch of inputs.

### v0.0.17 (2024-07-16)
*   **🚀 New Features**
    *   Added the `eval_mod()` function for modality evaluation.
    *   Added `skip_s` parameter to `init_model()` for more flexible model initialization.
*   **✨ Enhancements**
    *   Removed the deprecated `eval_scmib()` function.
*   **📚 Documentation**
    *   Added Tutorial 3, covering new evaluation methods.

### v0.0.16 (2024-07-11)
*   **🐛 Bug Fixes**
    *   Fixed an issue in `utils.load_predicted()` as reported in Issue #5.

### v0.0.15 (2024-07-11)
*   **🐛 Bug Fixes**
    *   Fixed an issue in the `reduce_data()` function.
    *   Corrected the sorting logic in `utils.ref_sort()` as reported in Issue #9.

### v0.0.14 (2024-07-04)
*   **✨ Enhancements**
    *   Improved compatibility and performance on **Windows** operating systems.
    *   Enhanced functionality for environments **without GPU support**.

### v0.0.13 (2024-07-04)
*   **🚀 New Features**
    *   Introduced `scmidas.datasets.GenDataFromPath()` for more flexible data input from custom paths.
    *   Added `viz_diff()` and `viz_mod_latent()` for advanced visualizations.
    *   Added new evaluation functions.
*   **✨ Enhancements**
    *   Renamed `pack()` to `reduce_data()` for better clarity.
*   **🐛 Bug Fixes**
    *   Addressed several minor bugs to improve stability.
*   **⚙️ Miscellaneous**
    *   Upgraded the minimum required Python version from `3.8` to `>=3.9` to accommodate `scib` dependencies.
*   **📚 Documentation**
    *   Updated all tutorials to align with the latest API changes.

### v0.0.8 (2024-06-20)
*   **🎉 Initial Release**
    *   First public version of the project.