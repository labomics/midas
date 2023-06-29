source("/root/workspace/code/midas/preprocess/utils.R")
setwd("/root/workspace/code/midas/")
library(RColorBrewer)
library(patchwork)

parser <- ArgumentParser()
parser$add_argument("--task", type = "character", default = "dogma_single_rna")
parser$add_argument("--experiment", type = "character", default = "e0")
parser$add_argument("--model", type = "character", default = "l_2")
parser$add_argument("--init_model", type = "character", default = "sp_00001999")
o <- parser$parse_known_args()[[1]]

pp_dir <- pj("data", "processed", o$task)
output_dir <- pj("result", "comparison", o$task, paste0("all_", o$task))
mkdir(output_dir, remove_old = F)
mod <- strsplit(o$task, "_")[[1]][3]

l <- 7.5  # figure size
L <- 10   # figure size
m <- 0.5  # legend margin

dirs <- list(
    "midas_embed"          = pj("result", "comparison", o$task, "midas_embed", o$experiment, o$model, o$init_model),
    "seurat_cca+wnn"       = pj("result", "comparison", o$task, "seurat_cca+wnn"),
    "seurat_rpca+wnn"      = pj("result", "comparison", o$task, "seurat_rpca+wnn")
)

plt_c_b <- NULL
plt_c_l <- NULL

for (method in names(dirs)) {
    message(paste0("Plotting ", method))
    dir <- dirs[[method]]
    if (method == "midas_embed") {
        obj <- LoadH5Seurat(pj(dir, "obj.h5seurat"), reductions = c("c.umap", "u.umap"))
    } else {
        obj <- LoadH5Seurat(pj(dir, "obj.h5seurat"), assays = mod, reductions = "umap")
    }

    if (method == "midas_embed") {
        p1 <- dim_plot(obj, w = L, h = L, reduction = "c.umap", no_axes = T, return_plt = T, display = F, border = T, raster = T,
            split.by = NULL, group.by = "batch", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_4, legend = F,
            save_path = pj(output_dir, paste(method, o$experiment, o$model, o$init_model, "c_merged_batch", sep = "_")))

        p2 <- dim_plot(obj, w = L, h = L, reduction = "c.umap", no_axes = T, return_plt = T, display = F, border = T, raster = T,
            split.by = NULL, group.by = "l1", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_8, legend = F,
            save_path = pj(output_dir, paste(method, o$experiment, o$model, o$init_model, "c_merged_label", sep = "_")))
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
w <- L * 3 + 1.38 * 2
h <- L * 2
plt_size(w, h)
ggsave(plot = plt_c, file = pj(output_dir, paste(mod, o$experiment, o$model, o$init_model, "merged_c.png", sep = "_")), width = w, height = h, scale = 0.5, limitsize = F)
ggsave(plot = plt_c, file = pj(output_dir, paste(mod, o$experiment, o$model, o$init_model, "merged_c.pdf", sep = "_")), width = w, height = h, scale = 0.5, limitsize = F)
