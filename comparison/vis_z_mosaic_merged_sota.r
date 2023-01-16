source("/root/workspace/code/sc-transformer/preprocess/utils.R")
setwd("/root/workspace/code/sc-transformer/")
library(RColorBrewer)
library(patchwork)

parser <- ArgumentParser()
# parser$add_argument("--task", type = "character", default = "dogma")
parser$add_argument("--task", type = "character", default = "teadog")
# parser$add_argument("--method", type = "character", default = "midas_embed")
# parser$add_argument("--method", type = "character", default = "scmomat")
# parser$add_argument("--method", type = "character", default = "stabmap")
parser$add_argument("--method", type = "character", default = "scvaeit")
parser$add_argument("--experiment", type = "character", default = "e0")
parser$add_argument("--model", type = "character", default = "default")
parser$add_argument("--init_model", type = "character", default = "sp_00001899")
o <- parser$parse_known_args()[[1]]

output_dir <- pj("paper", "3")
mkdir(output_dir, remove_old = F)

l <- 7.5  # figure size
L <- 10   # figure size
m <- 0.5  # legend margin

dirs <- list()
for (t in c("full", "paired_full", "paired_abc", "paired_ab", "paired_ac", "paired_bc", "single_full", "single")) {
    task <- paste0(o$task, "_", t)
    if (o$method == "midas_embed") {
        dirs[[task]] <- pj("result", "comparison", task, o$method, o$experiment, o$init_model)
    } else {
        dirs[[task]] <- pj("result", "comparison", task, o$method)
    }
}


source("/root/workspace/code/sc-transformer/preprocess/utils.R")
plt_c_b <- NULL
plt_c_l <- NULL

reduc <- ifelse(o$method == "midas_embed", "c.umap", "umap")
for (task in names(dirs)) {
    message(paste0("Plotting ", task))
    dir <- dirs[[task]]

    obj <- LoadH5Seurat(pj(dir, "obj.h5seurat"), reductions = c(reduc))

    p1 <- dim_plot(obj, title = rename_task(task), w = L, h = L, reduction = reduc, no_axes = T, return_plt = T, display = F, border = T, raster = T,
        split.by = NULL, group.by = "batch", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_4, legend = F)

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
w <- L * 8 + 2.5
h <- L * 2 + 1
plt_size(w, h)
ggsave(plot = plt_c, file = pj(output_dir, paste(o$task, o$method, "mosaic_merged.png", sep = "_")),
    width = w, height = h, scale = 0.5, limitsize = F)
ggsave(plot = plt_c, file = pj(output_dir, paste(o$task, o$method, "mosaic_merged.pdf", sep = "_")),
    width = w, height = h, scale = 0.5, limitsize = F)
