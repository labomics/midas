source("/root/workspace/code/sc-transformer/preprocess/utils.R")

base_dir <- "data/raw/rna+adt/asap/PBMC_stimulation/cite_ctrl"
rna_path <- pj(base_dir, "GSM4732113_CD28_CD3_control_CITE_GEX")
adt_path <- pj(base_dir, "GSM4732114_CD28_CD3_control_CITE_ADT.tsv.gz")


output_dir <- pj(base_dir, "seurat")
mkdir(output_dir, remove_old = T)


# RNA
# load data
rna_counts <- Read10X(rna_path)
rna <- gen_rna(rna_counts)
VlnPlot(rna, c("nFeature_rna", "nCount_rna", "percent.mt"),
        pt.size = 0.001, ncol = 3) + NoLegend()
rna
rna <- subset(rna, subset =
    nFeature_rna > 500 & nFeature_rna < 6000 &
    nCount_rna > 600 & nCount_rna < 30000 &
    percent.mt < 10
)
VlnPlot(rna, c("nFeature_rna", "nCount_rna", "percent.mt"),
        pt.size = 0.001, ncol = 3) + NoLegend()
rna

# ADT
# load data
adt_counts <- t(read.table(file = adt_path, sep = "\t", header = TRUE, row.names = 1))
adt_counts <- adt_counts[, colnames(rna_counts)]
adt <- gen_adt(adt_counts)
# QC
VlnPlot(adt, c("nCount_adt"), pt.size = 0.001, ncol = 1, log = T) + NoLegend()
adt
adt <- subset(adt, subset = nCount_adt > 400 & nCount_adt < 15000)
VlnPlot(adt, c("nCount_adt"), pt.size = 0.001, ncol = 1, log = T) + NoLegend()
adt


# Get intersected cells satisfying QC metrics of all modalities
cell_ids <- Reduce(intersect, list(colnames(rna), colnames(adt)))
rna <- subset(rna, cells = cell_ids)
adt <- subset(adt, cells = cell_ids)
rna
adt


# preprocess and save data
preprocess(output_dir, rna = rna, adt = adt)