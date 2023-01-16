# MIDAS

&mdash;Instruction for reproducing the manuscript results.

## Contents

- [MIDAS](#midas)
  - [Contents](#contents)
  - [File description](#file-description)
  - [Developing and test environment](#developing-and-test-environment)
  - [Quality control for individual datasets (27 batches in total)](#quality-control-for-individual-datasets-27-batches-in-total)
  - [Mosaic integration](#mosaic-integration)
    - [Generating training data for MIDAS](#generating-training-data-for-midas)
    - [Training MIDAS](#training-midas)
    - [Inferring latent variables, and generating imputed and batch corrected counts](#inferring-latent-variables-and-generating-imputed-and-batch-corrected-counts)
    - [Evaluation](#evaluation)
      - [Mosaic integration with complete modalities (rectangular integration)](#mosaic-integration-with-complete-modalities-rectangular-integration)
      - [Mosaic integration with missing modalities](#mosaic-integration-with-missing-modalities)
  - [Atlas construction](#atlas-construction)
    - [Generating training data for MIDAS](#generating-training-data-for-midas-1)
    - [Training MIDAS](#training-midas-1)
    - [Inferring latent variables, and generating imputed and batch corrected counts](#inferring-latent-variables-and-generating-imputed-and-batch-corrected-counts-1)
    - [Evaluation](#evaluation-1)
  - [Knowledge transfer](#knowledge-transfer)
    - [Reference construction](#reference-construction)
      - [Generating training data for MIDAS](#generating-training-data-for-midas-2)
      - [Training MIDAS](#training-midas-2)
      - [Inferring latent variables, and generating imputed and batch corrected counts](#inferring-latent-variables-and-generating-imputed-and-batch-corrected-counts-2)
    - [Mosaic integration with less modalities](#mosaic-integration-with-less-modalities)
      - [Generating training data for MIDAS](#generating-training-data-for-midas-3)
      - [Training MIDAS](#training-midas-3)
      - [Inferring latent variables, and generating imputed and batch corrected counts](#inferring-latent-variables-and-generating-imputed-and-batch-corrected-counts-3)
    - [Model transfer](#model-transfer)
      - [Generating training data for MIDAS](#generating-training-data-for-midas-4)
      - [Training MIDAS](#training-midas-4)
      - [Inferring latent variables, and generating imputed and batch corrected counts](#inferring-latent-variables-and-generating-imputed-and-batch-corrected-counts-4)
      - [Evaluation](#evaluation-2)
    - [Label transfer](#label-transfer)
      - [Reciprocal reference mapping for query dogma mosaic datasets](#reciprocal-reference-mapping-for-query-dogma-mosaic-datasets)
      - [Inferring latent variables, and generating imputed and batch corrected counts](#inferring-latent-variables-and-generating-imputed-and-batch-corrected-counts-5)
      - [Evaluation](#evaluation-3)

## File description


| File or directory | Description                                                 |
| ------------------- | ------------------------------------------------------------- |
| `comparison/`     | Scripts for algorithm comparison and qualitative evaluation |
| `configs/`        | Dataset configuration and MIDAS model configuration         |
| `eval/`           | Scripts for quantitative evaluation                         |
| `functions/`      | PyTorch functions for MIDAS                                 |
| `modules/`        | PyTorch models and dataloader for MIDAS                     |
| `preprocess/`     | Scripts for data preprocessing                              |
| `utils/`          | Commonly used functions                                     |
| `README.md`       | This file                                                   |
| `run.py`          | Script for MIDAS training and inference                     |

## Developing and test environment

System: Linux Ubuntu 18.04

Programming language:

- Python 3.7.11
- R 4.1.1

Main packages:

- PyTorch 1.12.0
- Seurat 4.1.0
- Signac 1.6.0

## Quality control for individual datasets (27 batches in total)

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

## Mosaic integration

Here we only demonstrate mosaic integration on dogma datasets. For teadog datasets, one can simply replace `dogma` with `teadog` in the command lines.

### Generating training data for MIDAS

Generating eight different dogma mosaic datasets:

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

### Training MIDAS

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

### Inferring latent variables, and generating imputed and batch corrected counts

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

### Evaluation

To generate ground-truth cell type labels for both qualitative and quantitative evaluation, we employed the third-party tool, Seurat, to annotate cell types for different PBMC datasets through label transfer.

```bash
Rscript preprocess/annotate_seurat.R
```

#### Mosaic integration with complete modalities (rectangular integration)

Performing rectangular integration with the state-of-the-art (SOTA) methods:

```bash
task=dogma_full
init=sp_00001899
Rscript comparison/midas_embed.r         --exp e0 --init_model $init --task $task &
Rscript comparison/harmony+wnn.r         --exp e0 --init_model $init --task $task &
Rscript comparison/pca+wnn.r             --exp e0 --init_model $init --task $task &
Rscript comparison/seurat_cca+wnn.r      --exp e0 --init_model $init --task $task &
Rscript comparison/seurat_rpca+wnn.r     --exp e0 --init_model $init --task $task &
Rscript comparison/scanorama_embed+wnn.r --exp e0 --init_model $init --task $task &
Rscript comparison/scanorama_feat+wnn.r  --exp e0 --init_model $init --task $task &
Rscript comparison/liger+wnn.r           --exp e0 --init_model $init --task $task &
Rscript comparison/mofa.r                --exp e0 --init_model $init --task $task &
Rscript comparison/bbknn.r               --exp e0 --init_model $init --task $task &
```

UMAP visualization of the joint embeddings generated by MIDAS and SOTA methods in rectangular integration:

```bash
Rscript comparison/vis_z_rect_sota.r --task dogma_full
```

Benchmarking rectangular integration with scIB:

```bash
task=dogma_full
init=sp_00001899
py eval/benchmark_batch_bio.py --task $task --init_model $init --method midas_embed &
py eval/benchmark_batch_bio.py --task $task --init_model $init --method harmony+wnn &
py eval/benchmark_batch_bio.py --task $task --init_model $init --method pca+wnn &
py eval/benchmark_batch_bio.py --task $task --init_model $init --method seurat_cca+wnn &
py eval/benchmark_batch_bio.py --task $task --init_model $init --method seurat_rpca+wnn &
py eval/benchmark_batch_bio.py --task $task --init_model $init --method scanorama_embed+wnn &
py eval/benchmark_batch_bio.py --task $task --init_model $init --method scanorama_feat+wnn &
py eval/benchmark_batch_bio.py --task $task --init_model $init --method liger+wnn &
py eval/benchmark_batch_bio.py --task $task --init_model $init --method mofa &
py eval/benchmark_batch_bio.py --task $task --init_model $init --method bbknn &
```

#### Mosaic integration with missing modalities

UMAP visualization of the joint embeddings generated by MIDAS in mosaic integration:

```bash
data=dogma
tasks=(full paired_full single_full paired_abc paired_ab paired_ac paired_bc single paired_a paired_b paired_c single_atac single_rna single_adt)
for task in "${tasks[@]}"; do
    Rscript comparison/midas_embed.r --exp e0 --init_model sp_00001899 --task $data"_"$task &
done
```

UMAP visualization of the modality embeddings and joint embeddings generated by MIDAS in mosaic integration:

```bash
data=dogma
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single paired_a paired_b paired_c single_atac single_rna single_adt)
for task in "${tasks[@]}"; do
    Rscript comparison/vis_z_mosaic_split.r --init_model sp_00001899 --task $data"_"$task &
done
```

Computing batch correction and biological conservation metrics of MIDAS on feature space (required by scMIB):

```bash
# generate wnn graph for evaluation
data=dogma
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single)
inits=(sp_00001899)
for init in "${inits[@]}"; do
    for i in "${!tasks[@]}"; do
        Rscript comparison/midas_feat+wnn.r --exp e0 --task $data"_"${tasks[$i]} --init_model $init &
    done
done
```

```bash
# evaluate using wnn graph
for init in "${inits[@]}"; do
    for i in "${!tasks[@]}"; do
        py eval/benchmark_batch_bio.py --method midas_feat+wnn --exp e0 --task $data"_"${tasks[$i]} --init_model $init &
    done
done
```

Computing batch correction and biological conservation metrics of MIDAS on embedding space (required by scIB and scMIB):

```bash
data=dogma
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single single_atac single_rna single_adt paired_a paired_b paired_c)
inits=(sp_00001899)
for init in "${inits[@]}"; do
    for i in "${!tasks[@]}"; do
        py eval/benchmark_batch_bio.py --method midas_embed --exp e0 --task $data"_"${tasks[$i]} --init_model $init &
    done
done
```

Computing modality alignment metrics of MIDAS (required by scMIB):

```bash
# generate ground-truth dogma_full datasets, which use the features of mosaic datasets
Rscript preprocess/combine_unseen.R --reference dogma_full        --task dogma_full_ref_full        && py preprocess/split_mat.py --task dogma_full_ref_full        &
Rscript preprocess/combine_unseen.R --reference dogma_paired_full --task dogma_full_ref_paired_full && py preprocess/split_mat.py --task dogma_full_ref_paired_full &
Rscript preprocess/combine_unseen.R --reference dogma_single_full --task dogma_full_ref_single_full && py preprocess/split_mat.py --task dogma_full_ref_single_full &
Rscript preprocess/combine_unseen.R --reference dogma_paired_abc  --task dogma_full_ref_paired_abc  && py preprocess/split_mat.py --task dogma_full_ref_paired_abc  &
Rscript preprocess/combine_unseen.R --reference dogma_paired_ab   --task dogma_full_ref_paired_ab   && py preprocess/split_mat.py --task dogma_full_ref_paired_ab   &
Rscript preprocess/combine_unseen.R --reference dogma_paired_ac   --task dogma_full_ref_paired_ac   && py preprocess/split_mat.py --task dogma_full_ref_paired_ac   &
Rscript preprocess/combine_unseen.R --reference dogma_paired_bc   --task dogma_full_ref_paired_bc   && py preprocess/split_mat.py --task dogma_full_ref_paired_bc   &
Rscript preprocess/combine_unseen.R --reference dogma_single      --task dogma_full_ref_single      && py preprocess/split_mat.py --task dogma_full_ref_single      &
```

```bash
# perform modality translation
data=dogma
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single)
inits=(sp_00001899)
act=translate
for init in "${inits[@]}"; do
    for i in "${!tasks[@]}"; do
        CUDA_VISIBLE_DEVICES=$i py run.py --task $data"_full_ref_"${tasks[$i]} --ref $data"_"${tasks[$i]} --act $act --init_model $init --init_from_ref 1 --exp e0 &
    done
done
```

```bash
# compute metrics
for init in "${inits[@]}"; do
    for i in "${!tasks[@]}"; do
        py eval/benchmark_mod.py --method midas_embed --exp e0 --task ${data}_${tasks[$i]} --init_model $init
    done
done
```

scMIB comparison of MIDAS mosaic integration results:

```bash
tasks="dogma_full dogma_paired_full dogma_paired_abc dogma_paired_ab dogma_paired_ac dogma_paired_bc dogma_single_full dogma_single"
py eval/combine_metrics_scmib.py --tasks $tasks --init_model sp_00001899
```

scIB comparison of MIDAS mosaic integration results and SOTA rectangular integration results:

```bash
tasks="dogma_full dogma_paired_full dogma_paired_abc dogma_paired_ab dogma_paired_ac dogma_paired_bc dogma_single_full dogma_single
       dogma_single_atac dogma_single_rna dogma_single_adt dogma_paired_a dogma_paired_b dogma_paired_c"
py eval/combine_metrics_scib.py --tasks $tasks --init_model sp_00001899 --mosaic 1 --sota 1
```

Performing mosaic integration with the SOTA methods:

```bash
data=dogma
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single)
for i in "${!tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=$i py comparison/scmomat1.py --task $data"_"${tasks[$i]} &
done
```

```bash
for i in "${!tasks[@]}"; do
    Rscript comparison/scmomat2.r --task $data"_"${tasks[$i]} &
    Rscript comparison/scvaeit.r --task $data"_"${tasks[$i]} &
    Rscript comparison/stabmap.r --task $data"_"${tasks[$i]} &
done
```

UMAP comparison of MIDAS and SOTA methods:

```bash
Rscript comparison/vis_z_mosaic_merged_sota.r --task dogma  --method midas_embed &
Rscript comparison/vis_z_mosaic_merged_sota.r --task dogma  --method scmomat &
Rscript comparison/vis_z_mosaic_merged_sota.r --task dogma  --method scvaeit &
Rscript comparison/vis_z_mosaic_merged_sota.r --task dogma  --method stabmap &
```

Computing batch correction and biological conservation metrics of SOTA methods on embedding space:

```bash
data=dogma
init=sp_00001899
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single)
for i in "${!tasks[@]}"; do
    py eval/benchmark_batch_bio.py --task $data"_"${tasks[$i]} --init_model $init --method scmomat --exp e0 &
    py eval/benchmark_batch_bio.py --task $data"_"${tasks[$i]} --init_model $init --method stabmap --exp e0 &
    py eval/benchmark_batch_bio.py --task $data"_"${tasks[$i]} --init_model $init --method scvaeit --exp e0 &
done
```

scIB comparison of the mosaic integration results for each SOTA method:

```bash
tasks="dogma_full dogma_paired_full dogma_paired_abc dogma_paired_ab dogma_paired_ac dogma_paired_bc dogma_single_full dogma_single"
init=sp_00001899
py eval/combine_metrics_scib.py --method scmomat --tasks $tasks --init_model $init --mosaic 1 --sota 0 &
py eval/combine_metrics_scib.py --method stabmap --tasks $tasks --init_model $init --mosaic 1 --sota 0 &
py eval/combine_metrics_scib.py --method scvaeit --tasks $tasks --init_model $init --mosaic 1 --sota 0 &
```

## Atlas construction

### Generating training data for MIDAS

```bash
Rscript preprocess/combine_subsets.R --task atlas && py preprocess/split_mat.py --task atlas &
```

### Training MIDAS

```bash
CUDA_VISIBLE_DEVICES=0 py run.py --exp e0 --task atlas
```

### Inferring latent variables, and generating imputed and batch corrected counts

```bash
CUDA_VISIBLE_DEVICES=0 py run.py --task atlas --act predict_all_latent_bc --init_model sp_latest &
```

### Evaluation

UMAP visualization of the joint embeddings generated by MIDAS in atlas construction:

```bash
Rscript comparison/vis_z_atlas.r --init_model sp_latest --task atlas 
```

## Knowledge transfer

### Reference construction

#### Generating training data for MIDAS

Generating reference dataset for evaluating knowledge transfer, where the dogma datasets are excluded:

```bash
Rscript preprocess/combine_subsets.R --task atlas_no_dogma  && py preprocess/split_mat.py --task atlas_no_dogma  &
```

#### Training MIDAS

```bash
CUDA_VISIBLE_DEVICES=5 py run.py --exp e0 --task atlas_no_dogma
```

#### Inferring latent variables, and generating imputed and batch corrected counts

```bash
CUDA_VISIBLE_DEVICES=1 py run.py --task atlas_no_dogma  --act predict_all_latent_bc --init_model sp_latest &
```

### Mosaic integration with less modalities

The goal of this experiment is to generate the baseline for transfer learning comparison.

#### Generating training data for MIDAS

```bash
Rscript preprocess/combine_subsets.R --task dogma_single_atac  && py preprocess/split_mat.py --task dogma_single_atac &
Rscript preprocess/combine_subsets.R --task dogma_single_rna   && py preprocess/split_mat.py --task dogma_single_rna &
Rscript preprocess/combine_subsets.R --task dogma_single_adt   && py preprocess/split_mat.py --task dogma_single_adt &
Rscript preprocess/combine_subsets.R --task dogma_paired_a     && py preprocess/split_mat.py --task dogma_paired_a  &
Rscript preprocess/combine_subsets.R --task dogma_paired_b     && py preprocess/split_mat.py --task dogma_paired_b  &
Rscript preprocess/combine_subsets.R --task dogma_paired_c     && py preprocess/split_mat.py --task dogma_paired_c  &
```

#### Training MIDAS

De novo training on the additional dogma mosaic datasets (for comparison):

```bash
CUDA_VISIBLE_DEVICES=0 py run.py --exp e0 --task dogma_single_atac &
CUDA_VISIBLE_DEVICES=1 py run.py --exp e0 --task dogma_single_rna &
CUDA_VISIBLE_DEVICES=2 py run.py --exp e0 --task dogma_single_adt &
CUDA_VISIBLE_DEVICES=3 py run.py --exp e0 --task dogma_paired_a &
CUDA_VISIBLE_DEVICES=4 py run.py --exp e0 --task dogma_paired_b &
CUDA_VISIBLE_DEVICES=5 py run.py --exp e0 --task dogma_paired_c &
```

#### Inferring latent variables, and generating imputed and batch corrected counts

```bash
CUDA_VISIBLE_DEVICES=0 py run.py --task dogma_single_atac --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=1 py run.py --task dogma_single_rna  --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=2 py run.py --task dogma_single_adt  --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=3 py run.py --task dogma_paired_a    --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=4 py run.py --task dogma_paired_b    --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
CUDA_VISIBLE_DEVICES=5 py run.py --task dogma_paired_c    --act predict_all_latent_bc --init_model sp_00001899 --exp e0 &
```

### Model transfer

#### Generating training data for MIDAS

Generating 14 different dogma mosaic datasets for querying:

```bash
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_full_transfer         && py preprocess/split_mat.py --task dogma_full_transfer        &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_full_transfer  && py preprocess/split_mat.py --task dogma_paired_full_transfer &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_single_full_transfer  && py preprocess/split_mat.py --task dogma_single_full_transfer &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_abc_transfer   && py preprocess/split_mat.py --task dogma_paired_abc_transfer  &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_ab_transfer    && py preprocess/split_mat.py --task dogma_paired_ab_transfer   &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_ac_transfer    && py preprocess/split_mat.py --task dogma_paired_ac_transfer   &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_bc_transfer    && py preprocess/split_mat.py --task dogma_paired_bc_transfer   &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_single_transfer       && py preprocess/split_mat.py --task dogma_single_transfer      &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_single_atac_transfer  && py preprocess/split_mat.py --task dogma_single_atac_transfer &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_single_rna_transfer   && py preprocess/split_mat.py --task dogma_single_rna_transfer  &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_single_adt_transfer   && py preprocess/split_mat.py --task dogma_single_adt_transfer  &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_a_transfer     && py preprocess/split_mat.py --task dogma_paired_a_transfer    &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_b_transfer     && py preprocess/split_mat.py --task dogma_paired_b_transfer    &
Rscript preprocess/combine_unseen.R --reference atlas_no_dogma --task dogma_paired_c_transfer     && py preprocess/split_mat.py --task dogma_paired_c_transfer    &
```

#### Training MIDAS

Transfer learning on query dogma mosaic datasets:

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

#### Inferring latent variables, and generating imputed and batch corrected counts

```bash
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single single_atac single_rna single_adt paired_a paired_b paired_c)
for i in "${!tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) py run.py --task "dogma_"${tasks[$i]}"_transfer" --ref atlas_no_dogma --act predict_all_latent_bc --init_model sp_00003699 &
done
```

#### Evaluation

UMAP visualization of the joint embeddings generated by MIDAS in mosaic integration:

```bash
data=dogma
tasks=(full paired_full single_full paired_abc paired_ab paired_ac paired_bc single paired_a paired_b paired_c single_atac single_rna single_adt)
for task in "${tasks[@]}"; do
    Rscript comparison/midas_embed.r --exp e0 --init_model sp_00003699 --task $data"_"$task"_transfer" &
done
```

UMAP visualization of the modality embeddings and joint embeddings generated by MIDAS in mosaic integration:

```bash
data=dogma
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single paired_a paired_b paired_c single_atac single_rna single_adt)
for task in "${tasks[@]}"; do
    Rscript comparison/vis_z_mosaic_split.r --init_model sp_00003699 --task $data"_"$task"_transfer" &
done
```

Computing batch correction and biological conservation metrics of MIDAS on feature space (required by scMIB):

```bash
# generate wnn graph for evaluation
data=dogma
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single)
inits=(sp_00003699)
for init in "${inits[@]}"; do
    for i in "${!tasks[@]}"; do
        Rscript comparison/midas_feat+wnn.r --exp e0 --task $data"_"${tasks[$i]}"_transfer" --init_model $init &
    done
done
```

```bash
# evaluate using wnn graph
for init in "${inits[@]}"; do
    for i in "${!tasks[@]}"; do
        py eval/benchmark_batch_bio.py --method midas_feat+wnn --exp e0 --task $data"_"${tasks[$i]}"_transfer" --init_model $init &
    done
done
```

Computing batch correction and biological conservation metrics of MIDAS on embedding space (required by scIB and scMIB):

```bash
data=dogma
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single single_atac single_rna single_adt paired_a paired_b paired_c)
inits=(sp_00003699)
for init in "${inits[@]}"; do
    for i in "${!tasks[@]}"; do
        py eval/benchmark_batch_bio.py --method midas_embed --exp e0 --task $data"_"${tasks[$i]}"_transfer" --init_model $init &
    done
done
```

Computing modality alignment metrics of MIDAS (required by scMIB):

```bash
# enerate ground-truth dogma_full datasets for scMIB modality alignment evaluation, which use the features of mosaic datasets
Rscript preprocess/combine_unseen.R --reference dogma_full_transfer        --task dogma_full_ref_full_transfer        && py preprocess/split_mat.py --task dogma_full_ref_full_transfer        &
Rscript preprocess/combine_unseen.R --reference dogma_paired_full_transfer --task dogma_full_ref_paired_full_transfer && py preprocess/split_mat.py --task dogma_full_ref_paired_full_transfer &
Rscript preprocess/combine_unseen.R --reference dogma_single_full_transfer --task dogma_full_ref_single_full_transfer && py preprocess/split_mat.py --task dogma_full_ref_single_full_transfer &
Rscript preprocess/combine_unseen.R --reference dogma_paired_abc_transfer  --task dogma_full_ref_paired_abc_transfer  && py preprocess/split_mat.py --task dogma_full_ref_paired_abc_transfer  &
Rscript preprocess/combine_unseen.R --reference dogma_paired_ab_transfer   --task dogma_full_ref_paired_ab_transfer   && py preprocess/split_mat.py --task dogma_full_ref_paired_ab_transfer   &
Rscript preprocess/combine_unseen.R --reference dogma_paired_ac_transfer   --task dogma_full_ref_paired_ac_transfer   && py preprocess/split_mat.py --task dogma_full_ref_paired_ac_transfer   &
Rscript preprocess/combine_unseen.R --reference dogma_paired_bc_transfer   --task dogma_full_ref_paired_bc_transfer   && py preprocess/split_mat.py --task dogma_full_ref_paired_bc_transfer   &
Rscript preprocess/combine_unseen.R --reference dogma_single_transfer      --task dogma_full_ref_single_transfer      && py preprocess/split_mat.py --task dogma_full_ref_single_transfer      &
```

```bash
# perform modality translation
data=dogma
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single)
inits=(sp_00003699)
act=translate
for init in "${inits[@]}"; do
    for i in "${!tasks[@]}"; do
        CUDA_VISIBLE_DEVICES=$i py run.py --task $data"_full_ref_"${tasks[$i]}"_transfer" --ref $data"_"${tasks[$i]}"_transfer" --act $act --init_model $init --init_from_ref 1 --exp e0 &
    done
done
```

```bash
# compute metrics
for init in "${inits[@]}"; do
    for i in "${!tasks[@]}"; do
        py eval/benchmark_mod.py --method midas_embed --exp e0 --task ${data}_${tasks[$i]}_transfer --init_model $init
    done
done
```

scMIB comparison of MIDAS mosaic integration results:

```bash
tasks="dogma_full_transfer dogma_paired_full_transfer dogma_paired_abc_transfer dogma_paired_ab_transfer dogma_paired_ac_transfer dogma_paired_bc_transfer dogma_single_full_transfer dogma_single_transfer"
py eval/combine_metrics_scmib.py --tasks $tasks --init_model sp_00003699
```

scIB comparison of MIDAS mosaic integration results and SOTA rectangular integration results:

```bash
tasks="dogma_full_transfer dogma_paired_full_transfer dogma_paired_abc_transfer dogma_paired_ab_transfer dogma_paired_ac_transfer dogma_paired_bc_transfer dogma_single_full_transfer dogma_single_transfer
       dogma_single_atac_transfer dogma_single_rna_transfer dogma_single_adt_transfer dogma_paired_a_transfer dogma_paired_b_transfer dogma_paired_c_transfer"
py eval/combine_metrics_scib.py --tasks $tasks --init_model sp_00003699 --mosaic 1 --sota 1
```

### Label transfer

#### Reciprocal reference mapping for query dogma mosaic datasets

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

#### Inferring latent variables, and generating imputed and batch corrected counts

```bash
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single single_atac single_rna single_adt paired_a paired_b paired_c)
for i in "${!tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=$((i % 8)) py run.py --task "dogma_"${tasks[$i]}"_continual" --ref atlas_no_dogma --act predict_joint --init_model sp_00003799 &
done
```

#### Evaluation

Performing query-to-reference mapping for comparison:

```bash
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single single_atac single_rna single_adt paired_a paired_b paired_c)
for i in "${!tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=$i py run.py --task "dogma_"${tasks[$i]}"_transfer" --ref atlas_no_dogma --init_from_ref 1 --act predict_joint --init_model sp_00001999 --drop_s 1 &
done
```

Computing the micro F1-scores of different mapping strategies:

```bash
# query-to-reference mapping
for i in "${!tasks[@]}"; do
    py eval/benchmark_transfer.py --task "dogma_"${tasks[$i]}"_transfer" --init_model sp_00001999 &
done
```

```bash
# reference-to-query mapping
for i in "${!tasks[@]}"; do
    py eval/benchmark_transfer.py --task "dogma_"${tasks[$i]}"_transfer" --init_model sp_00003699 &
done
```

```bash
# reciprocal reference mapping
for i in "${!tasks[@]}"; do
    py eval/benchmark_continual.py --task "dogma_"${tasks[$i]}"_continual" --ref atlas_no_dogma --init_model sp_00003799 &
done
```
