# Release notes

## V 0.1.X

**v0.1.8 2025-6-12**

1. Fixed a bug preventing the optimizer from being initialized after MIDAS.load_checkpoint().
2. Added support for .mtx input format.
3. Updated documentation and tutorials.
4. Add returns for MIDAS.predict(). Improve effeciency.
5. Add UMAP visualization during training. See MIDAS.configure_data_from_dir(viz_umap_tb=True).
6. Add data.download_models().
7. Update demo data.


**v0.1.7 2025-1-22**

1. fix bug: #22

**v0.1.6 2025-1-20**

1. fix bug: add condition for dims_h. Only when len(dims_x['atac'])>1, we use the 'dims_brefore_enc_atac' configuration.

**v0.1.5 2025-1-17**

1. fix bug: remove gaussian sampling during inferring for modality-specific embeddings.

**v0.1.4 2024-12-31**

1. debug: MIDAS.get_emb_umap(), correct load_predicted()

**v0.1.3 2024-12-21**

1. Integrate with Lightning to enable multi-GPU training.
2. Integrate with TensorBoard to facilitate loss visualization.
3. Enhance MIDAS to support easier integration of new modalities.

## V 0.0.X

**v0.0.18 2024-07-29**

1. utils.viz_mod_latent(). Rotated the image and fixed the bug that caused an error when inputting a batch.

**v0.0.17  2024-07-16**

1. add eval_mod(), remove eval_scmib()
2. add tutorial-3
3. add skip_s in init_model()

**v0.0.16  2024-07-11**

1. Fix utils.load_predicted(). See #5.

**v0.0.15  2024-07-11**

1. Fix reduce_data()
2. Fix utils.ref_sort(). See #9.

**v0.0.14  2024-07-04**

1. Adaptation for Windows: Improved compatibility and performance on Windows operating systems.
2. Non-GPU Compatibility: Enhanced functionality for environments without GPU support.

**v0.0.13  2024-07-04**

1. Python=3.8 -> Python>=3.9 to accomodate scib.
2. Add scmidas.datasets.GenDataFromPath() for flexible inputs.
3. Update tutorials.
4. Change pack() -> reduce_data().
5. Fix bugs.
6. Add viz_diff().
7. Add viz_mod_latent().
8. Add evaluation funcs.

**v0.0.8  2024-06-20**

First release.
