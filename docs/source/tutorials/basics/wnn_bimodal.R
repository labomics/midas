
args <- commandArgs(trailingOnly = TRUE)
temp_dir <- args[1]

library('Seurat')
library('dplyr')

print("R script: Reading data...")
rna <- read.csv(paste0(temp_dir, 'rna.csv'), header=TRUE, row.names=1)
adt <- read.csv(paste0(temp_dir, 'adt.csv'), header=TRUE, row.names=1)

print("R script: Creating Seurat object...")
obj <- CreateSeuratObject(counts = rna, assay = "rna")
obj[["adt"]] <- CreateAssayObject(counts = adt)
obj <- subset(obj, subset = nCount_rna > 0 & nCount_adt > 0)
print(obj)

print("R script: Running RNA processing...")
DefaultAssay(obj) <- 'rna'
VariableFeatures(obj) <- rownames(obj)
obj <-  NormalizeData(obj) %>%
        ScaleData() %>%
        RunPCA(reduction.name = "pca_rna", verbose = F)

print("R script: Running ADT processing...")
DefaultAssay(obj) <- 'adt'
VariableFeatures(obj) <- rownames(obj)
obj <-  NormalizeData(obj, normalization.method = "CLR", margin = 2) %>%
        ScaleData() %>%
        RunPCA(reduction.name = "pca_adt", verbose = F)

print("R script: Running WNN...")
obj <- FindMultiModalNeighbors(obj, list("pca_rna", "pca_adt"), list(1:32, 1:32))

print("R script: Running UMAP...")
obj <- RunUMAP(obj, nn.name = "weighted.nn", reduction.name = "umap")
umap_coords <- obj@reductions$umap@cell.embeddings
write.csv(umap_coords, file=paste0(temp_dir, 'umap_coords.csv'))
print("R script: Finished.")
