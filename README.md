# MIDAS

---Instructions for (1) the usage of code and (2) the reproduction of manuscript results (will be finished soon).

## Contents

- [MIDAS](#midas)
  - [Contents](#contents)
  - [File description](#file-description)
  - [Developing and test environment](#developing-and-test-environment)
  - [Data preprocessing](#data-preprocessing)
    - [Quality control for individual datasets (27 batches in total)](#quality-control-for-individual-datasets-27-batches-in-total)
    - [Generating training data for MIDAS](#generating-training-data-for-midas)
      - [Mosaic integration experiment](#mosaic-integration-experiment)
      - [Atlas construction experiment](#atlas-construction-experiment)
      - [Knowledge transfer experiment](#knowledge-transfer-experiment)
  - [Training MIDAS](#training-midas)
    - [Mosaic integration experiment](#mosaic-integration-experiment-1)
    - [Atlas construction experiment](#atlas-construction-experiment-1)
    - [Model transfer experiment](#model-transfer-experiment)
    - [Label transfer experiment](#label-transfer-experiment)
  - [Inferring latent variables, and generating imputed and batch corrected counts](#inferring-latent-variables-and-generating-imputed-and-batch-corrected-counts)
    - [Mosaic integration experiment](#mosaic-integration-experiment-2)
  - [Reproduction of the manuscript's results](#reproduction-of-the-manuscripts-results)

## File description

| File or directory | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| `comparison/`   | Scripts for algorithm comparison and qualitative evaluation |
| `configs/`      | Dataset configuration and MIDAS model configuration         |
| `eval/`         | Scripts for quantitative evaluation                         |
| `functions/`    | PyTorch functions for MIDAS                                 |
| `modules/`      | PyTorch models and dataloader for MIDAS                     |
| `preprocess/`   | Scripts for data preprocessing                              |
| `utils/`        | Commonly used functions                                     |
| `README.md`     | This file                                                   |
| `run.py`        | Script for MIDAS training and inference                     |

## Developing and test environment

System: Linux Ubuntu 18.04

Programming language:

- Python 3.7.11
- R 4.1.1

Main packages:

- PyTorch 1.12.0
- Seurat 4.1.0
- Signac 1.6.0

## Data preprocessing

### Quality control for individual datasets (27 batches in total)

```bash
# dogma (4 batches)
Rscript preprocess/atac+rna+adt_dogma_lll_ctrl.R &
Rscript preprocess/atac+rna+adt_dogma_lll_stim.R &
Rscript preprocess/atac+rna+adt_dogma_dig_ctrl.R &
Rscript preprocess/atac+rna+adt_dogma_dig_stim.R &

# tea (5 batches)
Rscript preprocess/atac+rna+adt_tea_w1.R &
Rscript preprocess/atac+rna+adt_tea_w3.R &
Rscript preprocess/atac+rna+adt_tea_w4.R &
Rscript preprocess/atac+rna+adt_tea_w5.R &
Rscript preprocess/atac+rna+adt_tea_w6.R &

### tea multiome (2 batches)
Rscript preprocess/atac+rna_tea_multiome_w1.R &
Rscript preprocess/atac+rna_tea_multiome_w2.R &

### 10x multiome (4 batches)
Rscript preprocess/atac+rna_10x_multiome_arc2_3k.R &
Rscript preprocess/atac+rna_10x_multiome_arc2_10k.R &
Rscript preprocess/atac+rna_10x_multiome_chrom_c_10k.R &
Rscript preprocess/atac+rna_10x_multiome_chrom_x_10k.R &

# asap (2 batches)
Rscript preprocess/atac+adt_asap_ctrl.R &
Rscript preprocess/atac+adt_asap_stim.R &

### asap cite (2 batches)
Rscript preprocess/rna+adt_asap_cite_ctrl.R &
Rscript preprocess/rna+adt_asap_cite_stim.R &

### wnn cite (8 batches)
Rscript preprocess/rna+adt_wnn.R &
```

### Generating training data for MIDAS

#### Mosaic integration experiment

Dogma mosaic datasets

```bash
Rscript preprocess/combine_subsets.R --task dogma_full        && py preprocess/split_mat.py --task dogma_full &
Rscript preprocess/combine_subsets.R --task dogma_paired_full && py preprocess/split_mat.py --task dogma_paired_full &
Rscript preprocess/combine_subsets.R --task dogma_single_full && py preprocess/split_mat.py --task dogma_single_full &
Rscript preprocess/combine_subsets.R --task dogma_paired_abc  && py preprocess/split_mat.py --task dogma_paired_abc &
Rscript preprocess/combine_subsets.R --task dogma_paired_ab   && py preprocess/split_mat.py --task dogma_paired_ab &
Rscript preprocess/combine_subsets.R --task dogma_paired_ac   && py preprocess/split_mat.py --task dogma_paired_ac &
Rscript preprocess/combine_subsets.R --task dogma_paired_bc   && py preprocess/split_mat.py --task dogma_paired_bc &
Rscript preprocess/combine_subsets.R --task dogma_single      && py preprocess/split_mat.py --task dogma_single &
```

Ground-truth dogma_full datasets for scMIB modality alignment evaluation, which use the features of mosaic datasets

```bash
Rscript preprocess/combine_unseen.R --reference dogma_full        --task dogma_full_ref_full        && py preprocess/split_mat.py --task dogma_full_ref_full        &
Rscript preprocess/combine_unseen.R --reference dogma_paired_full --task dogma_full_ref_paired_full && py preprocess/split_mat.py --task dogma_full_ref_paired_full &
Rscript preprocess/combine_unseen.R --reference dogma_single_full --task dogma_full_ref_single_full && py preprocess/split_mat.py --task dogma_full_ref_single_full &
Rscript preprocess/combine_unseen.R --reference dogma_paired_abc  --task dogma_full_ref_paired_abc  && py preprocess/split_mat.py --task dogma_full_ref_paired_abc  &
Rscript preprocess/combine_unseen.R --reference dogma_paired_ab   --task dogma_full_ref_paired_ab   && py preprocess/split_mat.py --task dogma_full_ref_paired_ab   &
Rscript preprocess/combine_unseen.R --reference dogma_paired_ac   --task dogma_full_ref_paired_ac   && py preprocess/split_mat.py --task dogma_full_ref_paired_ac   &
Rscript preprocess/combine_unseen.R --reference dogma_paired_bc   --task dogma_full_ref_paired_bc   && py preprocess/split_mat.py --task dogma_full_ref_paired_bc   &
Rscript preprocess/combine_unseen.R --reference dogma_single      --task dogma_full_ref_single      && py preprocess/split_mat.py --task dogma_full_ref_single      &
```

Teadog mosaic datasets

```bash
Rscript preprocess/combine_subsets.R --task teadog_full        && py preprocess/split_mat.py --task teadog_full &
Rscript preprocess/combine_subsets.R --task teadog_paired_full && py preprocess/split_mat.py --task teadog_paired_full &
Rscript preprocess/combine_subsets.R --task teadog_single_full && py preprocess/split_mat.py --task teadog_single_full &
Rscript preprocess/combine_subsets.R --task teadog_paired_abc  && py preprocess/split_mat.py --task teadog_paired_abc &
Rscript preprocess/combine_subsets.R --task teadog_paired_ab   && py preprocess/split_mat.py --task teadog_paired_ab &
Rscript preprocess/combine_subsets.R --task teadog_paired_ac   && py preprocess/split_mat.py --task teadog_paired_ac &
Rscript preprocess/combine_subsets.R --task teadog_paired_bc   && py preprocess/split_mat.py --task teadog_paired_bc &
Rscript preprocess/combine_subsets.R --task teadog_single      && py preprocess/split_mat.py --task teadog_single &
```

Ground-truth teadog_full datasets for scMIB modality alignment evaluation, which use the features of mosaic datasets

```bash
Rscript preprocess/combine_unseen.R --reference teadog_full        --task teadog_full_ref_full        && py preprocess/split_mat.py --task teadog_full_ref_full        &
Rscript preprocess/combine_unseen.R --reference teadog_paired_full --task teadog_full_ref_paired_full && py preprocess/split_mat.py --task teadog_full_ref_paired_full &
Rscript preprocess/combine_unseen.R --reference teadog_single_full --task teadog_full_ref_single_full && py preprocess/split_mat.py --task teadog_full_ref_single_full &
Rscript preprocess/combine_unseen.R --reference teadog_paired_abc  --task teadog_full_ref_paired_abc  && py preprocess/split_mat.py --task teadog_full_ref_paired_abc  &
Rscript preprocess/combine_unseen.R --reference teadog_paired_ab   --task teadog_full_ref_paired_ab   && py preprocess/split_mat.py --task teadog_full_ref_paired_ab   &
Rscript preprocess/combine_unseen.R --reference teadog_paired_ac   --task teadog_full_ref_paired_ac   && py preprocess/split_mat.py --task teadog_full_ref_paired_ac   &
Rscript preprocess/combine_unseen.R --reference teadog_paired_bc   --task teadog_full_ref_paired_bc   && py preprocess/split_mat.py --task teadog_full_ref_paired_bc   &
Rscript preprocess/combine_unseen.R --reference teadog_single      --task teadog_full_ref_single      && py preprocess/split_mat.py --task teadog_full_ref_single      &
```

#### Atlas construction experiment

Full atlas

```bash
Rscript preprocess/combine_subsets.R --task atlas           && py preprocess/split_mat.py --task atlas           &
```

#### Knowledge transfer experiment

Reference atlas for evaluating knowledge transfer, where the dogma datasets are excluded

```bash
Rscript preprocess/combine_subsets.R --task atlas_no_dogma  && py preprocess/split_mat.py --task atlas_no_dogma  &
```

Query dogma mosaic datasets

```bash
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_full_transfer        && py preprocess/split_mat.py --task dogma_full_transfer        &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_full_transfer && py preprocess/split_mat.py --task dogma_paired_full_transfer &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_single_full_transfer && py preprocess/split_mat.py --task dogma_single_full_transfer &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_abc_transfer  && py preprocess/split_mat.py --task dogma_paired_abc_transfer  &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_ab_transfer   && py preprocess/split_mat.py --task dogma_paired_ab_transfer   &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_ac_transfer   && py preprocess/split_mat.py --task dogma_paired_ac_transfer   &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_bc_transfer   && py preprocess/split_mat.py --task dogma_paired_bc_transfer   &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_single_transfer      && py preprocess/split_mat.py --task dogma_single_transfer      &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_single_atac_transfer  && py preprocess/split_mat.py --task dogma_single_atac_transfer &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_single_rna_transfer   && py preprocess/split_mat.py --task dogma_single_rna_transfer  &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_single_adt_transfer   && py preprocess/split_mat.py --task dogma_single_adt_transfer  &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_a_transfer     && py preprocess/split_mat.py --task dogma_paired_a_transfer  &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_b_transfer     && py preprocess/split_mat.py --task dogma_paired_b_transfer  &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_c_transfer     && py preprocess/split_mat.py --task dogma_paired_c_transfer  &
```

Additional dogma mosaic datasets for de novo training, which are used to generate the baseline for transfer learning comparison

```bash
Rscript preprocess/combine_subsets.R --task dogma_single_atac  && py preprocess/split_mat.py --task dogma_single_atac &
Rscript preprocess/combine_subsets.R --task dogma_single_rna   && py preprocess/split_mat.py --task dogma_single_rna &
Rscript preprocess/combine_subsets.R --task dogma_single_adt   && py preprocess/split_mat.py --task dogma_single_adt &
Rscript preprocess/combine_subsets.R --task dogma_paired_a     && py preprocess/split_mat.py --task dogma_paired_a  &
Rscript preprocess/combine_subsets.R --task dogma_paired_b     && py preprocess/split_mat.py --task dogma_paired_b  &
Rscript preprocess/combine_subsets.R --task dogma_paired_c     && py preprocess/split_mat.py --task dogma_paired_c  &
```

Ground-truth dogma_full datasets for scMIB modality alignment evaluation, which use the features of mosaic datasets

```bash
Rscript preprocess/combine_unseen.R --reference dogma_full_transfer        --task dogma_full_ref_full_transfer        && py preprocess/split_mat.py --task dogma_full_ref_full_transfer        &
Rscript preprocess/combine_unseen.R --reference dogma_paired_full_transfer --task dogma_full_ref_paired_full_transfer && py preprocess/split_mat.py --task dogma_full_ref_paired_full_transfer &
Rscript preprocess/combine_unseen.R --reference dogma_single_full_transfer --task dogma_full_ref_single_full_transfer && py preprocess/split_mat.py --task dogma_full_ref_single_full_transfer &
Rscript preprocess/combine_unseen.R --reference dogma_paired_abc_transfer  --task dogma_full_ref_paired_abc_transfer  && py preprocess/split_mat.py --task dogma_full_ref_paired_abc_transfer  &
Rscript preprocess/combine_unseen.R --reference dogma_paired_ab_transfer   --task dogma_full_ref_paired_ab_transfer   && py preprocess/split_mat.py --task dogma_full_ref_paired_ab_transfer   &
Rscript preprocess/combine_unseen.R --reference dogma_paired_ac_transfer   --task dogma_full_ref_paired_ac_transfer   && py preprocess/split_mat.py --task dogma_full_ref_paired_ac_transfer   &
Rscript preprocess/combine_unseen.R --reference dogma_paired_bc_transfer   --task dogma_full_ref_paired_bc_transfer   && py preprocess/split_mat.py --task dogma_full_ref_paired_bc_transfer   &
Rscript preprocess/combine_unseen.R --reference dogma_single_transfer      --task dogma_full_ref_single_transfer      && py preprocess/split_mat.py --task dogma_full_ref_single_transfer      &
```

## Training MIDAS

### Mosaic integration experiment

Train on dogma mosaic datasets

```bash
CUDA_VISIBLE_DEVICES=0 py run.py --exp e0 --task dogma_full &
CUDA_VISIBLE_DEVICES=1 py run.py --exp e0 --task dogma_paired_full &
CUDA_VISIBLE_DEVICES=2 py run.py --exp e0 --task dogma_single_full &
CUDA_VISIBLE_DEVICES=3 py run.py --exp e0 --task dogma_paired_abc &
CUDA_VISIBLE_DEVICES=4 py run.py --exp e0 --task dogma_paired_ab &
CUDA_VISIBLE_DEVICES=5 py run.py --exp e0 --task dogma_paired_ac &
CUDA_VISIBLE_DEVICES=6 py run.py --exp e0 --task dogma_paired_bc &
CUDA_VISIBLE_DEVICES=7 py run.py --exp e0 --task dogma_single &
```

Train on teadog mosaic datasets

```bash
CUDA_VISIBLE_DEVICES=0 py run.py --exp e0 --task teadog_full &
CUDA_VISIBLE_DEVICES=1 py run.py --exp e0 --task teadog_paired_full &
CUDA_VISIBLE_DEVICES=2 py run.py --exp e0 --task teadog_single_full &
CUDA_VISIBLE_DEVICES=3 py run.py --exp e0 --task teadog_paired_abc &
CUDA_VISIBLE_DEVICES=4 py run.py --exp e0 --task teadog_paired_ab &
CUDA_VISIBLE_DEVICES=5 py run.py --exp e0 --task teadog_paired_ac &
CUDA_VISIBLE_DEVICES=6 py run.py --exp e0 --task teadog_paired_bc &
CUDA_VISIBLE_DEVICES=7 py run.py --exp e0 --task teadog_single &
```

### Atlas construction experiment

Train on the full atlas

```bash
CUDA_VISIBLE_DEVICES=4 py run.py --exp e0 --task atlas
```

### Model transfer experiment

Pre-train on the reference atlas

```bash
CUDA_VISIBLE_DEVICES=5 py run.py --exp e0 --task atlas_no_dogma
```

Transfer learn on query dogma mosaic datasets

```bash
CUDA_VISIBLE_DEVICES=0 py run.py --task dogma_full_transfer         --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=1 py run.py --task dogma_paired_full_transfer  --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=2 py run.py --task dogma_paired_abc_transfer   --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=3 py run.py --task dogma_paired_ab_transfer    --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=4 py run.py --task dogma_paired_ac_transfer    --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=5 py run.py --task dogma_paired_bc_transfer    --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=6 py run.py --task dogma_single_full_transfer  --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=7 py run.py --task dogma_single_transfer       --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=0 py run.py --task dogma_single_atac_transfer  --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=1 py run.py --task dogma_single_rna_transfer   --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=2 py run.py --task dogma_single_adt_transfer   --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=0 py run.py --task dogma_paired_a_transfer     --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=1 py run.py --task dogma_paired_b_transfer     --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=2 py run.py --task dogma_paired_c_transfer     --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
```

De novo train on the additional dogma mosaic datasets (for comparison)

```bash
CUDA_VISIBLE_DEVICES=2 py run.py --exp e0 --task dogma_single_atac &
CUDA_VISIBLE_DEVICES=3 py run.py --exp e0 --task dogma_single_rna &
CUDA_VISIBLE_DEVICES=4 py run.py --exp e0 --task dogma_single_adt &
CUDA_VISIBLE_DEVICES=5 py run.py --exp e0 --task dogma_paired_a &
CUDA_VISIBLE_DEVICES=6 py run.py --exp e0 --task dogma_paired_b &
CUDA_VISIBLE_DEVICES=7 py run.py --exp e0 --task dogma_paired_c &
```

### Label transfer experiment

Reciprocal reference mapping for query dogma mosaic datasets

```bash
CUDA_VISIBLE_DEVICES=0 py run.py --task dogma_full_continual        --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=1 py run.py --task dogma_paired_full_continual --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=2 py run.py --task dogma_paired_abc_continual  --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=0 py run.py --task dogma_paired_ab_continual   --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=1 py run.py --task dogma_paired_ac_continual   --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=2 py run.py --task dogma_paired_bc_continual   --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=0 py run.py --task dogma_single_full_continual --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=1 py run.py --task dogma_single_continual      --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=2 py run.py --task dogma_single_atac_continual --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=0 py run.py --task dogma_single_rna_continual  --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=1 py run.py --task dogma_single_adt_continual  --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=0 py run.py --task dogma_paired_a_continual    --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=1 py run.py --task dogma_paired_b_continual    --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
CUDA_VISIBLE_DEVICES=2 py run.py --task dogma_paired_c_continual    --ref atlas_no_dogma --init_model sp_latest --init_from_ref 1 --epoch 4000 &
```

## Inferring latent variables, and generating imputed and batch corrected counts

### Mosaic integration experiment

Mosaic integration on dogma datasets

```bash
CUDA_VISIBLE_DEVICES=0 py run.py --task dogma_full        --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=1 py run.py --task dogma_paired_full --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=2 py run.py --task dogma_single_full --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=3 py run.py --task dogma_paired_abc  --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=4 py run.py --task dogma_paired_ab   --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=5 py run.py --task dogma_paired_ac   --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=6 py run.py --task dogma_paired_bc   --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=7 py run.py --task dogma_single      --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
```

Mosaic integration on teadog datasets

```bash
CUDA_VISIBLE_DEVICES=0 py run.py --task teadog_full        --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=1 py run.py --task teadog_paired_full --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=2 py run.py --task teadog_single_full --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=3 py run.py --task teadog_paired_abc  --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=4 py run.py --task teadog_paired_ab   --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=5 py run.py --task teadog_paired_ac   --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=6 py run.py --task teadog_paired_bc   --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=7 py run.py --task teadog_single      --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
```

## Reproduction of the manuscript's results


