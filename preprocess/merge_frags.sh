#!/bin/bash

cd data/raw/atac+rna/sea/BM_CD34/

# decompress files and add the same cell prefix as was added to the Seurat object
gzip -dc GSM6005303_BM_CD34_Rep1_atac_fragments.tsv.gz | awk 'BEGIN {FS=OFS="\t"} {print $1,$2,$3,$4"_1",$5}' - > cd34_rep1_fragments.tsv &
gzip -dc GSM6005305_BM_CD34_Rep2_atac_fragments.tsv.gz | awk 'BEGIN {FS=OFS="\t"} {print $1,$2,$3,$4"_2",$5}' - > cd34_rep2_fragments.tsv &

# merge files and sort
sort -m cd34_rep1_fragments.tsv cd34_rep2_fragments.tsv > cd34_fragments_.tsv
sort -k 1,1V -k2,2n cd34_fragments_.tsv > cd34_fragments.tsv
rm -rf cd34_fragments_.tsv

# block gzip compress the merged file
bgzip -@ 32 cd34_fragments.tsv # -@ 32 uses 32 threads




cd ../BM_Tcelldep/

# decompress files and add the same cell prefix as was added to the Seurat object
gzip -dc GSM6005307_BM_Tcelldep_Rep1_atac_fragments.tsv.gz | awk 'BEGIN {FS=OFS="\t"} {print $1,$2,$3,$4"_1",$5}' - > tcelldep_rep1_fragments.tsv &
gzip -dc GSM6005309_BM_Tcelldep_Rep2_atac_fragments.tsv.gz | awk 'BEGIN {FS=OFS="\t"} {print $1,$2,$3,$4"_2",$5}' - > tcelldep_rep2_fragments.tsv &

# merge files and sort
sort -m tcelldep_rep1_fragments.tsv tcelldep_rep2_fragments.tsv > tcelldep_fragments_.tsv
sort -k 1,1V -k2,2n tcelldep_fragments_.tsv > tcelldep_fragments.tsv
rm -rf tcelldep_fragments_.tsv

# block gzip compress the merged file
bgzip -@ 32 tcelldep_fragments.tsv # -@ 32 uses 32 threads