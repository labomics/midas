# Release Notes

All notable changes to this project will be documented in this file.

---

## Version 0.1.x

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