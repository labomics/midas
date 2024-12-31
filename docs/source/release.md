# Release notes

## v0.1.4 2024-12-21

1. debug: MIDAS.get_emb_umap(), correct load_predicted()

## v0.1.3 2024-12-21

1. Integrate with Lightning to enable multi-GPU training.
2. Integrate with TensorBoard to facilitate loss visualization.
3. Enhance MIDAS to support easier integration of new modalities.

## v0.0.18 2024-07-29

1. utils.viz_viz_mod_latent(). Rotated the image and fixed the bug that caused an error when inputting a batch.

## v0.0.17  2024-07-16

1. add eval_mod(), remove eval_scmib()
2. add tutorial-3
3. add skip_s in init_model()

## v0.0.16  2024-07-11

1. Fix utils.load_predicted(). See #5.

## v0.0.15  2024-07-11

1. Fix reduce_data()
2. Fix utils.ref_sort(). See #9.

## v0.0.14  2024-07-04

1. Adaptation for Windows: Improved compatibility and performance on Windows operating systems.
2. Non-GPU Compatibility: Enhanced functionality for environments without GPU support.

## v0.0.13  2024-07-04

1. Python=3.8 -> Python>=3.9 to accomodate scib.
2. Add scmidas.datasets.GenDataFromPath() for flexible inputs.
3. Update tutorials.
4. Change pack() -> reduce_data().
5. Fix bugs.
6. Add viz_diff().
7. Add viz_mod_latent().
8. Add evaluation funcs.

## v0.0.8  2024-06-20

First release.
