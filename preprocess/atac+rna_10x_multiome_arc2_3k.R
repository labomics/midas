source("/root/workspace/code/midas/preprocess/utils.R")

base_dir <- "data/raw/atac+rna/10x_multiome_arc2_3k"
count_path <- pj(base_dir, "pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5")
frag_path <- pj(base_dir, "pbmc_granulocyte_sorted_3k_atac_fragments.tsv.gz")


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
    nFeature_rna > 300 & nFeature_rna < 6000 &
    nCount_rna > 500 & nCount_rna < 2e4 &
    percent.mt < 20
)
VlnPlot(rna, c("nFeature_rna", "nCount_rna", "percent.mt"),
        pt.size = 0.001, ncol = 3) + NoLegend()
rna


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
cell_ids <- Reduce(intersect, list(colnames(atac), colnames(rna)))
atac <- subset(atac, cells = cell_ids)
rna <- subset(rna, cells = cell_ids)
atac
rna


# preprocess and save data
preprocess(output_dir, atac = atac, rna = rna)