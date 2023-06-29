source("/root/workspace/code/midas/preprocess/utils.R")
setwd("/root/workspace/code/midas/")
library(RColorBrewer)

parser <- ArgumentParser()
parser$add_argument("--task", type = "character", default = "dogma_single_adt")
parser$add_argument("--method", type = "character", default = "midas_embed")
parser$add_argument("--experiment", type = "character", default = "e0")
parser$add_argument("--model", type = "character", default = "default")
parser$add_argument("--init_model", type = "character", default = "sp_00001899")
o <- parser$parse_known_args()[[1]]

config <- parseTOML("configs/data.toml")[[gsub("_vd.*|_vt.*|_transfer$|_ref_.*$", "", o$task)]]
subset_names <- basename(config$raw_data_dirs)
subset_ids <- sapply(seq_along(subset_names) - 1, toString)
input_dirs <- pj("result", o$task, o$experiment, o$model, "predict", o$init_model, paste0("subset_", subset_ids))
pp_dir <- pj("data", "processed", o$task)
output_dir <- pj("result", "comparison", o$task, o$method, o$experiment, o$model, o$init_model)
mkdir(output_dir, remove_old = F)
if (grepl("_vt", o$task)) {
    fn <- paste0("l1_", tail(strsplit(o$task, split = "_")[[1]], 1), ".csv")
    label_paths <- pj(config$raw_data_dirs, "label_seurat", fn)
} else {
    label_paths <- pj(config$raw_data_dirs, "label_seurat", "l1.csv")
}

K <- parseTOML("configs/model.toml")[["default"]]$dim_c
l <- 7.5  # figure size
L <- 10   # figure size
m <- 0.6  # legend margin

z_list <- list()
cell_name_list <- list()
label_list <- list()
is_label <- T
subset_name_list <- list()
S <- length(subset_names)
for (i in seq_along(subset_names)) {
    subset_name <- subset_names[i]
    z_dir    <- pj(input_dirs[i], "z", "joint")
    fnames <- dir(path = z_dir, pattern = ".csv$")
    fnames <- str_sort(fnames, decreasing = F)

    z_subset_list <- list()
    N <- length(fnames)
    for (n in seq_along(fnames)) {
        message(paste0("Loading Subset ", i, "/", S, ", File ", n, "/", N))
        z_subset_list[[n]] <- read.csv(file.path(z_dir, fnames[n]), header = F)
    }
    z_list[[subset_name]] <- bind_rows(z_subset_list)

    cell_name_list[[subset_name]] <- read.csv(pj(pp_dir, paste0("subset_", subset_ids[i]),
        "cell_names.csv"), header = T)[, 2]
    if (file.exists(label_paths[i])) {
        label_list[[subset_name]] <- read.csv(label_paths[i], header = T)[, 2]
    } else {
        is_label <- F
    }
    
    subset_name_list[[subset_name]] <- rep(subset_name, length(cell_name_list[[subset_name]]))
}

rna <- t(data.matrix(bind_rows(z_list))) * 0  # pseudo rna counts
colnames(rna) <- do.call("c", unname(cell_name_list))
rownames(rna) <- paste0("rna-", seq_len(nrow(rna)))
obj <- CreateSeuratObject(counts = rna, assay = "rna")

z <- data.matrix(bind_rows(z_list))
c <- z[, 1:K]
colnames(c) <- paste0("c_", seq_len(ncol(c)))
rownames(c) <- colnames(obj)
obj[["c"]] <- CreateDimReducObject(embeddings = c, key = "c_", assay = "rna")

u <- z[, (K+1):(K+2)]
colnames(u) <- paste0("u_", seq_len(ncol(u)))
rownames(u) <- colnames(obj)
obj[["u"]] <- CreateDimReducObject(embeddings = u, key = "u_", assay = "rna")

obj@meta.data$batch <- factor(x = do.call("c", unname(subset_name_list)), levels = subset_names)
table(obj@meta.data$batch)[unique(obj@meta.data$batch)]
if (is_label) {
    obj@meta.data$l1 <- do.call("c", unname(label_list))
} else {
    obj@meta.data$l1 <- obj@meta.data$batch
}

obj

obj <- RunUMAP(obj, reduction = 'c', dims = 1:K, reduction.name = 'c.umap')
obj <- RunUMAP(obj, reduction = 'u', dims = 1:2, metric = "euclidean", reduction.name = 'u.umap')
SaveH5Seurat(obj, pj(output_dir, "obj.h5seurat"), overwrite = TRUE)


# obj <- LoadH5Seurat(pj(output_dir, "obj.h5seurat"), reductions = c("c.umap", "u.umap"))

dim_plot(obj, w = S*l, h = l+0.8, reduction = 'c.umap', no_axes = T, border = T, raster = T, raster_dpi = 500,
    split.by = "batch", group.by = "batch", label = F,
    repel = T, label.size = 4, pt.size = 1, cols = col_4,
    title = paste0(o$method, ", ", o$task), legend = F,
    save_path = pj(output_dir, paste(o$task, o$method, o$experiment, o$model, o$init_model, "c_split_batch", sep = "_")))

dim_plot(obj, w = S*l+m, h = l+0.8, reduction = 'c.umap', no_axes = T, border = T, raster = T, raster_dpi = 500,
    split.by = "batch", group.by = "l1", label = F,
    repel = T, label.size = 4, pt.size = 1, cols = col_8,
    title = paste0(o$method, ", ", o$task), legend = T,
    save_path = pj(output_dir, paste(o$task, o$method, o$experiment, o$model, o$init_model, "c_split_label", sep = "_")))

# dim_plot(obj, w = L+m, h = L, reduction = 'c.umap',
#     split.by = NULL, group.by = "batch", label = F,
#     repel = T, label.size = 4, pt.size = 0.1, cols = bcols,
#     title = o$method, legend = T,
#     save_path = pj(output_dir, paste(o$task, o$method, o$experiment, o$model, o$init_model, "c_merged_batch", sep = "_")))

# dim_plot(obj, w = L+m, h = L, reduction = 'c.umap',
#     split.by = NULL, group.by = "l1", label = F,
#     repel = T, label.size = 4, pt.size = 0.1, cols = dcols,
#     title = o$method, legend = T,
#     save_path = pj(output_dir, paste(o$task, o$method, o$experiment, o$model, o$init_model, "c_merged_label", sep = "_")))

# dim_plot(obj, w = L+m, h = L, reduction = 'u.umap',
#     split.by = NULL, group.by = "batch", label = F,
#     repel = T, label.size = 4, pt.size = 0.1, cols = bcols,
#     title = o$method, legend = T,
#     save_path = pj(output_dir, paste(o$task, o$method, o$experiment, o$model, o$init_model, "u_merged_batch", sep = "_")))

# dim_plot(obj, w = L+m, h = L, reduction = 'u.umap',
#     split.by = NULL, group.by = "l1", label = F,
#     repel = T, label.size = 4, pt.size = 0.1, cols = dcols,
#     title = o$method, legend = T,
#     save_path = pj(output_dir, paste(o$task, o$method, o$experiment, o$model, o$init_model, "u_merged_label", sep = "_")))


# obj <- LoadH5Seurat(pj(output_dir, "obj.h5seurat"), reductions = c("c.umap", "u.umap"))

dim_plot(obj, title = rename_task(o$task), w = L, h = L+m, reduction = 'c.umap', no_axes = T, border = T, raster = T, raster_dpi = 500,
    split.by = NULL, group.by = "batch", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_4, legend = F,
    save_path = pj(output_dir, paste(o$task, o$method, o$experiment, o$model, o$init_model, "c_merged_batch", sep = "_")))

dim_plot(obj, title = rename_task(o$task), w = L, h = L+m, reduction = 'c.umap', no_axes = T, border = T, raster = T, raster_dpi = 500,
    split.by = NULL, group.by = "l1", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_8, legend = F,
    save_path = pj(output_dir, paste(o$task, o$method, o$experiment, o$model, o$init_model, "c_merged_label", sep = "_")))

dim_plot(obj, title = rename_task(o$task), w = L, h = L+m, reduction = 'u.umap', no_axes = T, border = T, raster = T, raster_dpi = 500,
    split.by = NULL, group.by = "batch", label = F,  repel = T, label.size = 4, pt.size = 0.1, cols = col_4, legend = F,
    save_path = pj(output_dir, paste(o$task, o$method, o$experiment, o$model, o$init_model, "u_merged_batch", sep = "_")))

dim_plot(obj, title = rename_task(o$task), w = L, h = L+m, reduction = 'u.umap', no_axes = T, border = T, raster = T, raster_dpi = 500,
    split.by = NULL, group.by = "l1", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_8, legend = F,
    save_path = pj(output_dir, paste(o$task, o$method, o$experiment, o$model, o$init_model, "u_merged_label", sep = "_")))
