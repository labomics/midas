source("/root/workspace/code/midas/preprocess/utils.R")
setwd("/root/workspace/code/midas/")
library(gridExtra)
library(RColorBrewer)

parser <- ArgumentParser()
parser$add_argument("--task", type = "character", default = "dogma_single_atac")
parser$add_argument("--reduc", type = "character", default = "rpca")  # rpca or cca
parser$add_argument("--exp", type = "character", default = "e0")
parser$add_argument("--model", type = "character", default = "default")
parser$add_argument("--init_model", type = "character", default = "sp_00001899")
o <- parser$parse_known_args()[[1]]

o$method <- paste0("seurat_", o$reduc, "+wnn")
config <- parseTOML("configs/data.toml")[[o$task]]
subset_names <- basename(config$raw_data_dirs)
subset_ids <- sapply(seq_along(subset_names) - 1, toString)
input_dirs <- pj("result", o$task, o$exp, o$model, "predict", o$init_model, paste0("subset_", subset_ids))
pp_dir <- pj("data", "processed", o$task)
output_dir <- pj("result", "comparison", o$task, o$method)
mkdir(output_dir, remove_old = F)
label_paths <- pj(config$raw_data_dirs, "label_seurat", "l1.csv")

K <- parseTOML("configs/model.toml")[["default"]]$dim_c
l <- 7.5  # figure size
L <- 10   # figure size
m <- 0.5  # legend margin

rna_list <- list()
atac_list <- list()
adt_list <- list()
cell_name_list <- list()
label_list <- list()
subset_name_list <- list()
S <- length(subset_names)
for (i in seq_along(subset_names)) {
    subset_name <- subset_names[i]
    rna_dir  <- pj(input_dirs[i], "x", "rna")
    atac_dir <- pj(input_dirs[i], "x", "atac")
    adt_dir  <- pj(input_dirs[i], "x", "adt")
    fnames <- dir(path = rna_dir, pattern = ".csv$")
    if (length(fnames) == 0) {
        fnames <- dir(path = adt_dir, pattern = ".csv$")
        if (length(fnames) == 0) {
            fnames <- dir(path = atac_dir, pattern = ".csv$")
        }
    }
    fnames <- str_sort(fnames, decreasing = F)
    
    rna_subset_list <- list()
    atac_subset_list <- list()
    adt_subset_list <- list()
    N <- length(fnames)
    for (n in seq_along(fnames)) {
        message(paste0("Loading Subset ", i, "/", S, ", File ", n, "/", N))
        if (file.exists(file.path(rna_dir, fnames[n]))) {
            rna_subset_list[[n]] <- read.csv(file.path(rna_dir, fnames[n]), header = F)
        }
        if (file.exists(file.path(atac_dir, fnames[n]))) {
            atac_subset_list[[n]] <- read.csv(file.path(atac_dir, fnames[n]), header = F)
        }
        if (file.exists(file.path(adt_dir, fnames[n]))) {
            adt_subset_list[[n]] <- read.csv(file.path(adt_dir, fnames[n]), header = F)
        }
    }
    rna_list[[subset_name]] <- bind_rows(rna_subset_list)
    atac_list[[subset_name]] <- bind_rows(atac_subset_list)
    adt_list[[subset_name]] <- bind_rows(adt_subset_list)

    cell_name_list[[subset_name]] <- read.csv(pj(pp_dir, paste0("subset_", subset_ids[i]),
        "cell_names.csv"), header = T)[, 2]
    label_list[[subset_name]] <- read.csv(label_paths[i], header = T)[, 2]
    subset_name_list[[subset_name]] <- rep(subset_name, length(cell_name_list[[subset_name]]))
}

mods <- vector()
if (length(rna_list[[1]]) > 0) {
    mods <- c(mods, "rna")
}
if (length(adt_list[[1]]) > 0) {
    mods <- c(mods, "adt")
}
if (length(atac_list[[1]]) > 0) {
    mods <- c(mods, "atac")
}

cell_name <- do.call("c", unname(cell_name_list))
assays <- list()

if ("rna" %in% mods) {
    rna <- t(data.matrix(bind_rows(rna_list)))
    colnames(rna) <- cell_name
    rownames(rna) <- read.csv(pj(pp_dir, "feat", "feat_names_rna.csv"), header = T)[, 2]
    # remove missing features
    rna_mask_list <- list()
    for (i in seq_along(subset_names)) {
        subset_name <- subset_names[i]
        rna_mask_list[[subset_name]] <- read.csv(pj(pp_dir, paste0("subset_", subset_ids[i]),
            "mask", "rna.csv"), header = T)[, -1]
    }
    rna_mask <- as.logical(apply(data.matrix(bind_rows(rna_mask_list)), 2, prod))
    rna <- rna[rna_mask, ]
    assays[["rna"]] <- CreateAssayObject(counts = rna)
}

if ("adt" %in% mods) {
    adt <- t(data.matrix(bind_rows(adt_list)))
    colnames(adt) <- cell_name
    rownames(adt) <- read.csv(pj(pp_dir, "feat", "feat_names_adt.csv"), header = T)[, 2]
    # remove missing features
    adt_mask_list <- list()
    for (i in seq_along(subset_names)) {
        subset_name <- subset_names[i]
        adt_mask_list[[subset_name]] <- read.csv(pj(pp_dir, paste0("subset_", subset_ids[i]),
            "mask", "adt.csv"), header = T)[, -1]
    }
    adt_mask <- as.logical(apply(data.matrix(bind_rows(adt_mask_list)), 2, prod))
    adt <- adt[adt_mask, ]
    assays[["adt"]] <- CreateAssayObject(counts = adt)
}

if ("atac" %in% mods) {
    atac <- t(data.matrix(bind_rows(atac_list)))
    colnames(atac) <- cell_name
    rownames(atac) <- read.csv(pj(pp_dir, "feat", "feat_names_atac.csv"), header = T)[, 2]
    assays[["atac"]] <- CreateChromatinAssay(counts = atac)
}

first_mod <- T
for (mod in mods) {
    if (first_mod) {
        obj <- CreateSeuratObject(counts = assays[[mod]], assay = mod)
        first_mod <- F
    } else {
        obj[[mod]] <- assays[[mod]]
    }
}

obj@meta.data$l1 <- do.call("c", unname(label_list))
obj@meta.data$batch <- factor(x = do.call("c", unname(subset_name_list)), levels = subset_names)
table(obj@meta.data$batch)[unique(obj@meta.data$batch)]

obj
if ("rna" %in% mods) {
    obj <- subset(obj, subset = nCount_rna > 0)
}
if ("adt" %in% mods) {
    obj <- subset(obj, subset = nCount_adt > 0)
}
if ("atac" %in% mods) {
    obj <- subset(obj, subset = nCount_atac > 0)
}
obj

if ("rna" %in% mods) {
    obj_rna <- GetAssayData(object = obj, assay = "rna")
    obj_rna <- CreateSeuratObject(counts = obj_rna, assay = "rna")
    obj_rna@meta.data$l2 <- do.call("c", unname(label_list))
    obj_rna@meta.data$batch <- do.call("c", unname(subset_name_list))
    obj_rna.list <- SplitObject(obj_rna, split.by = "batch")
    obj_rna.list <- lapply(X = obj_rna.list, FUN = function(x) {
        x <- NormalizeData(x)
        x <- FindVariableFeatures(x, nfeatures = 5000)
    })

    rna_features <- SelectIntegrationFeatures(object.list = obj_rna.list, nfeatures = 5000)
    obj_rna.list <- lapply(X = obj_rna.list, FUN = function(x) {
        x <- ScaleData(x, features = rna_features, verbose = FALSE)
        x <- RunPCA(x, features = rna_features, verbose = FALSE, reduction.name = "pca")
    })
    rna.anchors <- FindIntegrationAnchors(
        object.list = obj_rna.list,
        anchor.features = rna_features,
        reduction = o$reduc)
    rna.combined <- IntegrateData(anchorset = rna.anchors)

    obj[["rna_int"]] <- GetAssay(rna.combined, assay = "integrated")
    DefaultAssay(obj) <- "rna_int"
    obj <- ScaleData(obj, verbose = FALSE)
    obj <- RunPCA(obj, reduction.name = paste0("pca_", o$reduc, "_rna"))
}

if ("adt" %in% mods) {
    obj_adt <- GetAssayData(object = obj, assay = "adt")
    obj_adt <- CreateSeuratObject(counts = obj_adt, assay = "adt")
    obj_adt@meta.data$l2 <- do.call("c", unname(label_list))
    obj_adt@meta.data$batch <- do.call("c", unname(subset_name_list))
    obj_adt.list <- SplitObject(obj_adt, split.by = "batch")
    obj_adt.list <- lapply(X = obj_adt.list, FUN = function(x) {
        x <- NormalizeData(x, normalization.method = "CLR", margin = 2)
        x <- FindVariableFeatures(x)
    })

    adt_features <- SelectIntegrationFeatures(object.list = obj_adt.list)
    obj_adt.list <- lapply(X = obj_adt.list, FUN = function(x) {
        x <- ScaleData(x, features = adt_features, verbose = FALSE)
        x <- RunPCA(x, features = adt_features, verbose = FALSE, reduction.name = "pca")
    })
    adt.anchors <- FindIntegrationAnchors(
        object.list = obj_adt.list,
        anchor.features = adt_features,
        reduction = o$reduc)
    adt.combined <- IntegrateData(anchorset = adt.anchors)

    obj[["adt_int"]] <- GetAssay(adt.combined, assay = "integrated")
    DefaultAssay(obj) <- "adt_int"
    obj <- ScaleData(obj, verbose = FALSE)
    obj <- RunPCA(obj, reduction.name = paste0("pca_", o$reduc, "_adt"))
}

if ("atac" %in% mods) {
    obj_atac <- GetAssayData(object = obj, assay = "atac")
    obj_atac <- CreateSeuratObject(counts = obj_atac, assay = "atac")
    obj_atac@meta.data$l2 <- do.call("c", unname(label_list))
    obj_atac@meta.data$batch <- do.call("c", unname(subset_name_list))
    obj_atac.list <- SplitObject(obj_atac, split.by = "batch")
    obj_atac.list <- lapply(X = obj_atac.list, FUN = function(x) {
        x <- RunTFIDF(x)
        x <- FindTopFeatures(x, min.cutof = "q25")
    })

    atac_features <- SelectIntegrationFeatures(object.list = obj_atac.list)
    obj_atac.list <- lapply(X = obj_atac.list, FUN = function(x) {
        x <- RunSVD(x, features = atac_features, verbose = FALSE, reduction.name = "lsi")
    })
    atac.anchors <- FindIntegrationAnchors(
        object.list = obj_atac.list,
        anchor.features = atac_features,
        reduction = ifelse(o$reduc == "rpca", "rlsi", o$reduc),
        dims = 2:K)
    atac.combined <- IntegrateData(anchorset = atac.anchors)

    obj[["atac_int"]] <- GetAssay(atac.combined, assay = "integrated")
    DefaultAssay(obj) <- "atac_int"
    obj <- RunSVD(obj, reduction.name = paste0("pca_", o$reduc, "_atac"))
}

# wnn
if (length(mods) == 3) {
    obj <- FindMultiModalNeighbors(obj, list(paste0("pca_", o$reduc, "_atac"), paste0("pca_", o$reduc, "_rna"), paste0("pca_", o$reduc, "_adt")),
                                        list(2:K, 1:K, 1:K))
    obj <- RunUMAP(obj, nn.name = "weighted.nn", reduction.name = "umap")
} else if (length(mods) == 1) {
    K0 <- ifelse(mods[[1]] == "atac", 2, 1)
    obj <- FindNeighbors(obj, paste0("pca_", o$reduc, "_", mods[[1]]), K0:K, graph.name = "wsnn")
    obj <- RunUMAP(obj, reduction = paste0("pca_", o$reduc, "_", mods[[1]]), dims = K0:K, reduction.name = 'umap')
} else {
    stop("Unsupported modality combination!")
}
SaveH5Seurat(obj, pj(output_dir, "obj.h5seurat"), overwrite = TRUE)

# save connectivity matrices for benchmarking
connectivities <- obj$wsnn
diag(connectivities) <- 0
invisible(writeMM(connectivities, pj(output_dir, "connectivities.mtx")))

# obj <- LoadH5Seurat(pj(output_dir, "obj.h5seurat"), assays = mods[[1]], reductions = "umap")

# dim_plot(obj, w = 4*l, h = l, reduction = "umap",
#     split.by = "batch", group.by = "batch", label = F,
#     repel = T, label.size = 4, pt.size = 0.5, cols = NULL,
#     title = o$method, legend = F,
#     save_path = pj(output_dir, paste0(o$method, "_split_batch")))

# dim_plot(obj, w = 4*l+m, h = l, reduction = "umap",
#     split.by = "batch", group.by = "l1", label = F,
#     repel = T, label.size = 4, pt.size = 0.5, cols = dcols,
#     title = o$method, legend = T,
#     save_path = pj(output_dir, paste0(o$method, "_split_label")))

# dim_plot(obj, w = L+m, h = L, reduction = "umap",
#     split.by = NULL, group.by = "batch", label = F,
#     repel = T, label.size = 4, pt.size = 0.1, cols = NULL,
#     title = o$method, legend = T,
#     save_path = pj(output_dir, paste0(o$method, "_merged_batch")))

# dim_plot(obj, w = L+m, h = L, reduction = "umap",
#     split.by = NULL, group.by = "l1", label = F,
#     repel = T, label.size = 4, pt.size = 0.1, cols = dcols,
#     title = o$method, legend = T,
#     save_path = pj(output_dir, paste0(o$method, "_merged_label")))

# obj <- LoadH5Seurat(pj(output_dir, "obj.h5seurat"), assays = mods[[1]], reductions = "umap")

dim_plot(obj, w = L, h = L, reduction = 'umap', no_axes = T,
    split.by = NULL, group.by = "batch", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_4, legend = F,
    save_path = pj(output_dir, paste(o$method, "merged_batch", sep = "_")))

dim_plot(obj, w = L, h = L, reduction = 'umap', no_axes = T,
    split.by = NULL, group.by = "l1", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_8, legend = F,
    save_path = pj(output_dir, paste(o$method, "merged_label", sep = "_")))
