source("/root/workspace/code/midas/preprocess/utils.R")
library(patchwork)


ref_dir <- "data/raw/rna+adt/wnn"
data_dirs <- c(
    "data/raw/atac+rna+adt/tea/TEA-seq/w1",
    "data/raw/atac+rna+adt/tea/TEA-seq/w3",
    "data/raw/atac+rna+adt/tea/TEA-seq/w4",
    "data/raw/atac+rna+adt/tea/TEA-seq/w5",
    "data/raw/atac+rna+adt/tea/TEA-seq/w6",
    "data/raw/atac+rna+adt/dogma/lll_ctrl",
    "data/raw/atac+rna+adt/dogma/lll_stim",
    "data/raw/atac+rna+adt/dogma/dig_ctrl",
    "data/raw/atac+rna+adt/dogma/dig_stim"
)


# Load reference data
# reference <- LoadH5Seurat(pj(ref_dir, "multi.h5seurat"))
reference <- LoadH5Seurat(pj(ref_dir, "multi.h5seurat")) %>%
             SCTransform(assay = "SCT", verbose = T)
# SaveH5Seurat(reference, pj(ref_dir, "multi_SCTransform.h5seurat"), overwrite = T)
# reference <- LoadH5Seurat(pj(ref_dir, "multi_SCTransform.h5seurat"))



p1 <- DimPlot(object = reference, reduction = "wnn.umap", group.by = "celltype.l1", label = T,
    label.size = 3, repel = T) + NoLegend()
p2 <- DimPlot(object = reference, reduction = "wnn.umap", group.by = "celltype.l2", label = T,
    label.size = 3, repel = T) + NoLegend()
p3 <- DimPlot(object = reference, reduction = "wnn.umap", group.by = "celltype.l3", label = T,
    label.size = 3, repel = T) + NoLegend()
p1 + p2 + p3
ggsave(file = file.path(ref_dir, "l1_l2_l3.png"), width = 18, height = 6)


for (data_dir in data_dirs) {
    query_dir <- pj(data_dir, "seurat")
    label_dir <- pj(data_dir, "label_seurat")
    mkdir(label_dir, remove_old = T)

    # Load query data
    query <- LoadH5Seurat(pj(query_dir, "rna.h5seurat")) %>%
             SCTransform(assay = "rna", verbose = T)
    # SaveH5Seurat(query, pj(query_dir, "rna_sct.h5seurat"), overwrite = T)
    # query <- LoadH5Seurat(pj(query_dir, "rna_sct.h5seurat"))

    # Find mapping
    anchors <- FindTransferAnchors(
        reference = reference,
        query = query,
        normalization.method = "SCT",
        reference.reduction = "spca",
        # recompute.residuals = F,
        dims = 1:50
    )

    # Transfer labels
    query <- MapQuery(
        anchorset = anchors,
        query = query,
        reference = reference,
        refdata = list(celltype.l1 = "celltype.l1",
                       celltype.l2 = "celltype.l2",
                       celltype.l3 = "celltype.l3",
                       predicted_ADT = "ADT"),
        reference.reduction = "spca",
        reduction.model = "wnn.umap"
    )
    write.csv(query@meta.data$predicted.celltype.l1, file = pj(label_dir, "l1.csv"))
    write.csv(query@meta.data$predicted.celltype.l2, file = pj(label_dir, "l2.csv"))
    write.csv(query@meta.data$predicted.celltype.l3, file = pj(label_dir, "l3.csv"))

    # Visualize
    p1 <- DimPlot(query, reduction = "ref.umap", group.by = "predicted.celltype.l1", label = T,
             label.size = 3, repel = T) + NoLegend()
    p2 <- DimPlot(query, reduction = "ref.umap", group.by = "predicted.celltype.l2", label = T,
             label.size = 3 ,repel = T) + NoLegend()
    p3 <- DimPlot(query, reduction = "ref.umap", group.by = "predicted.celltype.l3", label = T,
             label.size = 3 ,repel = T) + NoLegend()
    p1 + p2 + p3
    ggsave(file = file.path(label_dir, "ref_map.png"), width = 18, height = 6)
}