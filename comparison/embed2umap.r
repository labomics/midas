source("/root/workspace/code/midas/preprocess/utils.R")
setwd("/root/workspace/code/midas/")
library(RColorBrewer)

parser <- ArgumentParser()
parser$add_argument("--task", type = "character", default = "bm")
parser$add_argument("--method", type = "character", default = "scmomat")
parser$add_argument("--experiment", type = "character", default = "e0")
parser$add_argument("--model", type = "character", default = "default")
parser$add_argument("--init_model", type = "character", default = "sp_00001899")
o <- parser$parse_known_args()[[1]]

config <- parseTOML("configs/data.toml")[[gsub("_transfer$|_ref_.*$", "", o$task)]]
subset_names <- basename(config$raw_data_dirs)
subset_ids <- sapply(seq_along(subset_names) - 1, toString)
input_dirs <- pj("result", o$task, o$experiment, o$model, "predict", o$init_model, paste0("subset_", subset_ids))
pp_dir <- pj("data", "processed", o$task)
output_dir <- pj("result", "comparison", o$task, o$method)
mkdir(output_dir, remove_old = F)
label_paths <- pj(config$raw_data_dirs, "label_seurat", "l1.csv")

K <- parseTOML("configs/model.toml")[["default"]]$dim_c
l <- 7.5  # figure size
L <- 10   # figure size
m <- 0.5  # legend margin

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
        z_subset_list[[n]] <- read.csv(pj(z_dir, fnames[n]), header = F)
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

if (o$method %in% c("multigrate", "stabmap", "uniport", "glue", "scipenn", "totalvi")) {
    z <- data.matrix(read.csv(pj(output_dir, "embeddings.csv"), header = T, row.names = 1))
} else {
    z <- data.matrix(read.csv(pj(output_dir, "embeddings.csv"), header = F))
}

colnames(z) <- paste0("z_", seq_len(ncol(z)))
rownames(z) <- colnames(obj)
obj[["z"]] <- CreateDimReducObject(embeddings = z, key = "z_", assay = "rna")

obj@meta.data$l1 <- do.call("c", unname(label_list))
obj@meta.data$batch <- factor(x = do.call("c", unname(subset_name_list)), levels = subset_names)
table(obj@meta.data$batch)[unique(obj@meta.data$batch)]

obj


obj <- RunUMAP(obj, reduction = "z", dims = 1:K, reduction.name = "umap")
SaveH5Seurat(obj, pj(output_dir, "obj.h5seurat"), overwrite = TRUE)

# obj <- LoadH5Seurat(pj(output_dir, "obj.h5seurat"), assays = "adt", reductions = "umap")

dim_plot(obj, w = L, h = L, reduction = 'umap', no_axes = T,
    split.by = NULL, group.by = "batch", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_4, legend = F,
    save_path = pj(output_dir, paste(o$method, "merged_batch", sep = "_")))

dim_plot(obj, w = L, h = L, reduction = 'umap', no_axes = T,
    split.by = NULL, group.by = "l1", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_8, legend = F,
    save_path = pj(output_dir, paste(o$method, "merged_label", sep = "_")))

