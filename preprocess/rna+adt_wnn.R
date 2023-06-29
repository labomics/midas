source("/root/workspace/code/midas/preprocess/utils.R")

base_dir <- "data/raw/rna+adt/wnn"



obj <- LoadH5Seurat(pj(base_dir, "multi.h5seurat"))
obj_split <- SplitObject(obj, split.by = "orig.ident")
for (donor in unique(obj@meta.data$donor)) {
    batch <- paste0(donor, "_0")
    prt("Processing batch ", batch, " ...")
    output_dir <- pj(base_dir, tolower(batch), "seurat")
    mkdir(output_dir, remove_old = T)

    rna_counts <- obj_split[[batch]]$SCT@counts
    adt_counts <- obj_split[[batch]]$ADT@counts

    # RNA
    rna <- gen_rna(rna_counts)
    VlnPlot(rna, c("nFeature_rna", "nCount_rna", "percent.mt"),
            pt.size = 0.001, ncol = 3) + NoLegend()
    rna
    rna <- subset(rna, subset =
        nFeature_rna > 500 & nFeature_rna < 6000 &
        nCount_rna > 600 & nCount_rna < 40000 &
        percent.mt < 15
    )
    VlnPlot(rna, c("nFeature_rna", "nCount_rna", "percent.mt"),
            pt.size = 0.001, ncol = 3) + NoLegend()
    rna
    
    # ADT
    adt <- gen_adt(adt_counts)
    # QC
    VlnPlot(adt, c("nCount_adt"), pt.size = 0.001, ncol = 1, log = T) + NoLegend()
    adt
    adt <- subset(adt, subset = nCount_adt > 400 & nCount_adt < 20000)
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
}



