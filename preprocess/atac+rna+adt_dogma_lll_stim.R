source("./preprocess/utils.R")

base_dir <- "data/raw/atac+rna+adt/dogma/lll_stim"
frag_path <- pj(base_dir, "GSM5065527_LLL_STIM_fragments.tsv.gz")
count_path <- pj(base_dir, "GSM5065528_LLL_STIM_GExp_ATAC_filtered")
adt_path <- pj(base_dir, "GSM5065529_LLL_stim_ADT_allCounts.mtx.gz")
adt_bc_path <- pj(base_dir, "GSM5065529_LLL_stim_ADT_allCounts.barcodes.txt.gz")
adt_feat_path <- pj(base_dir, "GSM5065529_LLL_stim_ADT_allCounts.proteins.txt.gz")


output_dir <- pj(base_dir, "seurat")
mkdir(output_dir, remove_old = T)


# RNA
# load data
counts <- Read10X(count_path)
rna_counts <- counts$`Gene Expression`
atac_counts <- counts$`Peaks`
rna <- gen_rna(rna_counts)
VlnPlot(rna, c("nFeature_rna", "nCount_rna", "percent.mt"),
        pt.size = 0.001, ncol = 3) + NoLegend()
rna
rna <- subset(rna, subset =
    nFeature_rna > 300 & nFeature_rna < 6000 &
    nCount_rna > 500 & nCount_rna < 2e4 &
    percent.mt < 10
)
VlnPlot(rna, c("nFeature_rna", "nCount_rna", "percent.mt"),
        pt.size = 0.001, ncol = 3) + NoLegend()
rna


# ADT
# load data
adt_counts <- t(readMM(adt_path))  # D * N
adt_bc <- read.delim(adt_bc_path, header = F)
adt_feat <- read.delim(adt_feat_path, header = F)
colnames(adt_counts) <- paste0(adt_bc[, 1], "-1")
rownames(adt_counts) <- adt_feat[, 1]
adt_counts <- adt_counts[, colnames(rna_counts)]
adt <- gen_adt(adt_counts)
# QC
VlnPlot(adt, c("nCount_adt"), pt.size = 0.001, ncol = 1) + NoLegend()
adt
adt <- subset(adt, subset = nCount_adt > 400 & nCount_adt < 10000)
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