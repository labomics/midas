source("/root/workspace/code/sc-transformer/preprocess/utils.R")

base_dir <- "data/raw/atac+adt/asap/PBMC_stimulation/asap_ctrl"
frag_path <- pj(base_dir, "GSM4732109_CD28_CD3_control_ASAP_fragments.tsv.gz")
adt_path <- pj(base_dir, "GSM4732110_CD28_CD3_control_ASAP_ADT.tsv.gz")


output_dir <- pj(base_dir, "seurat")
mkdir(output_dir, remove_old = T)


# ADT
# load data
adt_counts <- t(read.table(file = adt_path, sep = "\t", header = TRUE, row.names = 1))
adt <- gen_adt(adt_counts)
# QC
VlnPlot(adt, c("nCount_adt"), pt.size = 0.001, ncol = 1, log = T) + NoLegend()
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
cell_ids <- Reduce(intersect, list(colnames(atac), colnames(adt)))
atac <- subset(atac, cells = cell_ids)
adt <- subset(adt, cells = cell_ids)
atac
adt


# preprocess and save data
preprocess(output_dir, atac = atac, adt = adt)