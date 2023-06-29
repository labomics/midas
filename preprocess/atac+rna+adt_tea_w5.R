source("/root/workspace/code/midas/preprocess/utils.R")

base_dir <- "data/raw/atac+rna+adt/tea/TEA-seq/w5"
frag_path <- pj(base_dir, "GSM5123953_X066-MP0C1W5_leukopak_perm-cells_tea_200M_atac_filtered_fragments.tsv.gz")
count_path <- pj(base_dir, "GSM5123953_X066-MP0C1W5_leukopak_perm-cells_tea_200M_cellranger-arc_filtered_feature_bc_matrix.h5")
adt_path <- pj(base_dir, "GSM5123953_X066-MP0C1W5_leukopak_perm-cells_tea_48M_adt_counts.csv.gz")


output_dir <- pj(base_dir, "seurat")
mkdir(output_dir, remove_old = T)


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
    nFeature_rna > 300 & nFeature_rna < 4000 &
    nCount_rna > 500 & nCount_rna < 1e4 &
    percent.mt < 60
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