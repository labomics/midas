#!/bin/bash

cd data/processed

data=dogma
tasks=(full paired_full paired_abc paired_ab paired_ac paired_bc single_full single single_atac single_rna single_adt paired_a paired_b paired_c)

# src_dir=$PWD
src_dir=/dev/shm/processed

for task in "${tasks[@]}"; do

    task=$data"_"$task
    mkdir -p ${task}_continual

    for i in {0..3}; do
        ln -sfn $src_dir/${task}_transfer/subset_$i $PWD/${task}_continual/subset_$i
    done

    ln -sfn $src_dir/atlas_no_$data/feat $PWD/${task}_continual/feat
    for i in {0..22}; do
        ln -sfn $src_dir/atlas_no_$data/subset_$i $PWD/${task}_continual/subset_$((i + 4))
    done

done


# data=dogma_lll
# tasks=(full paired_a paired_b paired_c atac rna adt)

# # src_dir=$PWD
# src_dir=/dev/shm/processed

# for task in "${tasks[@]}"; do

#     task=$data"_"$task
#     mkdir -p ${task}_continual

#     for i in {0..1}; do
#         ln -sfn $src_dir/${task}_transfer/subset_$i $PWD/${task}_continual/subset_$i
#     done

#     ln -sfn $src_dir/atlas_no_$data/feat $PWD/${task}_continual/feat
#     for i in {0..24}; do
#         ln -sfn $src_dir/atlas_no_$data/subset_$i $PWD/${task}_continual/subset_$((i + 2))
#     done

# done