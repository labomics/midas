source("/root/workspace/code/sc-transformer/preprocess/utils.R")

base_dir <- "data/raw/atac+rna+adt/tea/TEA-seq/w1"
frag_path <- pj(base_dir, "GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_fulldepth_atac_filtered_fragments.tsv.gz")
count_path <- pj(base_dir, "GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_fulldepth_cellranger-arc_filtered_feature_bc_matrix.h5")
adt_path <- pj(base_dir, "GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_fulldepth_adt_counts.csv.gz")


output_dir <- pj(base_dir, "seurat")
mkdir(output_dir, remove_old = T)

# frags <- read.table(frag_path)
# library(rhdf5)
# win <- h5ls(pj(base_dir, "GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_fulldepth_atac_window_20k.h5"))
# RNA
# load data
counts <- Read10X_h5(count_path)
rna_counts <- counts$`Gene Expression`
atac_counts <- counts$`Peaks`
rna <- gen_rna(rna_counts)
VlnPlot(rna, c("nFeature_rna", "nCount_rna", "percent.mt"),
        pt.size = 0.001, ncol = 3) + NoLegend()
rna
rna <- subset(rna, subset =
    nFeature_rna > 400 & nFeature_rna < 4000 &
    nCount_rna > 500 & nCount_rna < 13000 &
    percent.mt < 50
)
VlnPlot(rna, c("nFeature_rna", "nCount_rna", "percent.mt"),
        pt.size = 0.001, ncol = 3) + NoLegend()
rna


# ADT
# load data
adt_counts <- t(read.csv(file = adt_path, row.names = 1))[-1, ]
colnames(adt_counts) <- paste0(colnames(adt_counts), "-1")
adt_counts <- adt_counts[, colnames(rna_counts)]
adt <- gen_adt(adt_counts)
# QC
VlnPlot(adt, c("nCount_adt"), pt.size = 0.001, ncol = 1) + NoLegend()
adt
adt <- subset(adt, subset = nCount_adt > 400 & nCount_adt < 5000)
VlnPlot(adt, c("nCount_adt"), pt.size = 0.001, ncol = 1, log = T) + NoLegend()
adt


# ATAC
atac <- gen_atac(frag_path)
# QC
VlnPlot(atac, c("nCount_atac", "nucleosome_signal", "TSS.enrichment"),
        pt.size = 0.001, ncol = 3, log = T) + NoLegend()
atac
atac <- subset(atac, subset =
    nCount_atac > 500 & nCount_atac < 3e4 &
    nucleosome_signal < 2 &
    TSS.enrichment > 2
)
atac <- subset(atac, features = rownames(atac)[rowSums(atac$atac@counts > 0) > 5])
VlnPlot(atac, c("nCount_atac", "nucleosome_signal", "TSS.enrichment"),
        pt.size = 0.001, ncol = 3, log = F) + NoLegend()
atac
metadata <- read.csv(file = gsub("fragments.tsv", "metadata.csv", frag_path))
atac <- RenameCells(atac, new.names = metadata$original_barcodes[match(colnames(atac), metadata$barcodes)])


# Get intersected cells satisfying QC metrics of all modalities
cell_ids <- Reduce(intersect, list(colnames(atac), colnames(rna), colnames(adt)))
atac <- subset(atac, cells = cell_ids)
rna <- subset(rna, cells = cell_ids)
adt <- subset(adt, cells = cell_ids)
atac
rna
adt


# preprocess and save data
preprocess(output_dir, atac = atac, rna = rna, adt = adt)