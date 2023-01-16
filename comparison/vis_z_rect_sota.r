source("/root/workspace/code/sc-transformer/preprocess/utils.R")
setwd("/root/workspace/code/sc-transformer/")
library(RColorBrewer)
library(patchwork)

parser <- ArgumentParser()
parser$add_argument("--task", type = "character", default = "teadog_full")
parser$add_argument("--experiment", type = "character", default = "e0")
parser$add_argument("--model", type = "character", default = "default")
parser$add_argument("--init_model", type = "character", default = "sp_00001899")
o <- parser$parse_known_args()[[1]]

pp_dir <- pj("data", "processed", o$task)
output_dir <- pj("result", "comparison", o$task, paste0("all_", o$task))
mkdir(output_dir, remove_old = F)

l <- 7.5  # figure size
L <- 10   # figure size
m <- 0.5  # legend margin

dirs <- list(
    "midas_embed"          = pj("result", "comparison", o$task, "midas_embed", o$experiment, o$init_model),
    "midas_feat+wnn"       = pj("result", "comparison", o$task, "midas_feat+wnn", o$experiment, o$init_model),
    "bbknn"                = pj("result", "comparison", o$task, "bbknn"),
    "harmony+wnn"          = pj("result", "comparison", o$task, "harmony+wnn"),
    "liger+wnn"            = pj("result", "comparison", o$task, "liger+wnn"),
    "mofa"                 = pj("result", "comparison", o$task, "mofa"),
    "pca+wnn"              = pj("result", "comparison", o$task, "pca+wnn"),
    "scanorama_embed+wnn"  = pj("result", "comparison", o$task, "scanorama_embed+wnn"),
    "scanorama_feat+wnn"   = pj("result", "comparison", o$task, "scanorama_feat+wnn"),
    "seurat_cca+wnn"       = pj("result", "comparison", o$task, "seurat_cca+wnn"),
    "seurat_rpca+wnn"      = pj("result", "comparison", o$task, "seurat_rpca+wnn")
)

source("/root/workspace/code/sc-transformer/preprocess/utils.R")
plt_c_b <- NULL
plt_c_l <- NULL

for (method in names(dirs)) {
    message(paste0("Plotting ", method))
    dir <- dirs[[method]]
    if (method == "midas_embed") {
        obj <- LoadH5Seurat(pj(dir, "obj.h5seurat"), reductions = c("c.umap", "u.umap"))
    } else if (method == "scanorama_feat+wnn") {
        obj <- LoadH5Seurat(pj(dir, "obj.h5seurat"), assays = "adt_bc", reductions = "umap")
    } else {
        obj <- LoadH5Seurat(pj(dir, "obj.h5seurat"), assays = "adt", reductions = "umap")
    }

    if (method == "midas_embed") {
        p1 <- dim_plot(obj, w = L, h = L, reduction = "c.umap", no_axes = T, return_plt = T, display = F, border = T, raster = T,
            split.by = NULL, group.by = "batch", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_4, legend = F,
            save_path = pj(output_dir, paste(method, o$experiment, o$init_model, "c_merged_batch", sep = "_")))

        p2 <- dim_plot(obj, w = L, h = L, reduction = "c.umap", no_axes = T, return_plt = T, display = F, border = T, raster = T,
            split.by = NULL, group.by = "l1", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_8, legend = F,
            save_path = pj(output_dir, paste(method, o$experiment, o$init_model, "c_merged_label", sep = "_")))

        plt_u_b <- dim_plot(obj, w = L, h = L, reduction = "u.umap", no_axes = T, return_plt = T, display = F, border = T, raster = T,
            split.by = NULL, group.by = "batch", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_4, legend = F,
            save_path = pj(output_dir, paste(method, o$experiment, o$init_model, "u_merged_batch", sep = "_")))

        plt_u_l <- dim_plot(obj, w = L, h = L, reduction = "u.umap", no_axes = T, return_plt = T, display = F, border = T, raster = T,
            split.by = NULL, group.by = "l1", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_8, legend = F,
            save_path = pj(output_dir, paste(method, o$experiment, o$init_model, "u_merged_label", sep = "_")))
    } else if (method == "midas_feat+wnn") {
        p1 <- dim_plot(obj, w = L, h = L, reduction = "umap", no_axes = T, return_plt = T, display = F, border = T, raster = T,
            split.by = NULL, group.by = "batch", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_4, legend = F,
            save_path = pj(output_dir, paste(method, o$experiment, o$init_model, "merged_batch", sep = "_")))

        p2 <- dim_plot(obj, w = L, h = L, reduction = "umap", no_axes = T, return_plt = T, display = F, border = T, raster = T,
            split.by = NULL, group.by = "l1", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_8, legend = F,
            save_path = pj(output_dir, paste(method, o$experiment, o$init_model, "merged_label", sep = "_")))
    } else {
        p1 <- dim_plot(obj, w = L, h = L, reduction = "umap", no_axes = T, return_plt = T, display = F, border = T, raster = T,
            split.by = NULL, group.by = "batch", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_4, legend = F,
            save_path = pj(output_dir, paste(method, "merged_batch", sep = "_")))

        p2 <- dim_plot(obj, w = L, h = L, reduction = "umap", no_axes = T, return_plt = T, display = F, border = T, raster = T,
            split.by = NULL, group.by = "l1", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_8, legend = F,
            save_path = pj(output_dir, paste(method, "merged_label", sep = "_")))
    }

    p1 <- p1 + labs(colour = "Batch")
    p2 <- p2 + labs(colour = "Cell type")

    if (is.null(plt_c_b)) {
        plt_c_b <- p1
        plt_c_l <- p2
    } else {
        plt_c_b <- plt_c_b + p1
        plt_c_l <- plt_c_l + p2
    }
}

plt_c_b <- plt_c_b + plot_layout(nrow = 1, guides = "collect") & theme(legend.position = "right")
plt_c_l <- plt_c_l + plot_layout(nrow = 1, guides = "collect") & theme(legend.position = "right")
plt_c <- plt_c_b / plt_c_l
w <- L * 11 + 1.38 * 2
h <- L * 2
plt_size(w, h)
ggsave(plot = plt_c, file = pj(output_dir, "merged_c.png"), width = w, height = h, scale = 0.5, limitsize = F)
ggsave(plot = plt_c, file = pj(output_dir, "merged_c.pdf"), width = w, height = h, scale = 0.5, limitsize = F)


plt_u <- (plt_u_b + labs(colour = "Sample")    & theme(legend.position = "right")) /
         (plt_u_l + labs(colour = "Cell type") & theme(legend.position = "right"))
w <- L * 1 + 1.38 * 2
h <- L * 2
plt_size(w, h)
ggsave(plot = plt_u, file = pj(output_dir, "merged_u.png"), width = w, height = h, scale = 0.5, limitsize = F)
ggsave(plot = plt_u, file = pj(output_dir, "merged_u.pdf"), width = w, height = h, scale = 0.5, limitsize = F)

