source("/root/workspace/code/midas/preprocess/utils.R")
setwd("/root/workspace/code/midas/")
library(RColorBrewer)
library(patchwork)

parser <- ArgumentParser()
# parser$add_argument("--task", type = "character", default = "dogma")
parser$add_argument("--task", type = "character", default = "teadog")
parser$add_argument("--mods", type = "character", default = "rna+atac")
# parser$add_argument("--mods", type = "character", default = "rna+adt")
parser$add_argument("--method", type = "character", default = "midas_embed")
# parser$add_argument("--method", type = "character", default = "scmomat")
# parser$add_argument("--method", type = "character", default = "stabmap")
# parser$add_argument("--method", type = "character", default = "scvaeit")
# parser$add_argument("--method", type = "character", default = "multigrate")
# parser$add_argument("--method", type = "character", default = "cobolt")
# parser$add_argument("--method", type = "character", default = "multivi")
# parser$add_argument("--method", type = "character", default = "uniport")
parser$add_argument("--experiment", type = "character", default = "k_5")
parser$add_argument("--model", type = "character", default = "l_2")
parser$add_argument("--init_model", type = "character", default = "sp_00001899")
o <- parser$parse_known_args()[[1]]

output_dir <- pj("paper", "3")
mkdir(output_dir, remove_old = F)

l <- 7.5  # figure size
L <- 10   # figure size
m <- 0.5  # legend margin

dirs <- list()
if (o$mods == "rna+atac") {
    tasks <- c("paired_a", "rna_paired_a", "atac_paired_a", "diagonal_d_paired_a", "diagonal_b")
    if (o$task == "dogma") {
        init_models <- c("sp_00001799", "sp_00001899", "sp_00001699", "sp_00001899", "sp_00001999")
    } else {
        init_models <- c("sp_00001999", "sp_00001699", "sp_00001699", "sp_00001599", "sp_00001699")
    }
} else {
    tasks <- c("paired_c", "rna_paired_c", "adt_paired_c", "diagonal_c_paired_c", "diagonal_c")
    init_models <- c("sp_00001899", "sp_00001899", "sp_00001899", "sp_00001899", "sp_00001899")
}

for (i in seq_along(tasks)) {
    task <- paste0(o$task, "_", tasks[i])
    if (o$method == "midas_embed") {
        dirs[[task]] <- pj("result", "comparison", task, o$method, o$experiment, o$model, init_models[i])
    } else {
        dirs[[task]] <- pj("result", "comparison", task, o$method)
    }
}


source("/root/workspace/code/midas/preprocess/utils.R")
plt_c_b <- NULL
plt_c_l <- NULL

reduc <- ifelse(o$method == "midas_embed", "c.umap", "umap")
for (task in names(dirs)) {
    message(paste0("Plotting ", task))
    fp <- pj(dirs[[task]], "obj.h5seurat")
    if (file.exists(fp)) {
        obj <- LoadH5Seurat(fp, reductions = c(reduc))
    } else {
        obj <- NULL
    }
    p1 <- dim_plot(obj, title = rename_task_bimodal(task), w = L, h = L, reduction = reduc, no_axes = T, return_plt = T, display = F, border = T, raster = T,
        split.by = NULL, group.by = "batch", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_4, legend = F) +
        theme(plot.title = element_text(size = 34.5))
    p2 <- dim_plot(obj, w = L, h = L, reduction = reduc, no_axes = T, return_plt = T, display = F, border = T, raster = T,
        split.by = NULL, group.by = "l1", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_8, legend = F)
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
w <- L * 5 + 2.5
h <- L * 2 + 1
plt_size(w, h)
ggsave(plot = plt_c, file = pj(output_dir, paste(o$task, o$mods, o$method, "mosaic_merged.png", sep = "_")),
    width = w, height = h, scale = 0.5, limitsize = F)
ggsave(plot = plt_c, file = pj(output_dir, paste(o$task, o$mods, o$method, "mosaic_merged.pdf", sep = "_")),
    width = w, height = h, scale = 0.5, limitsize = F)
