source("/root/workspace/code/midas/preprocess/utils.R")
setwd("/root/workspace/code/midas/")
library(RColorBrewer)
library(patchwork)

parser <- ArgumentParser()
parser$add_argument("--task", type = "character", default = "bm")
parser$add_argument("--experiment", type = "character", default = "e0")
parser$add_argument("--model", type = "character", default = "la_1")
parser$add_argument("--init_model", type = "character", default = "sp_00001899")
o <- parser$parse_known_args()[[1]]

pp_dir <- pj("data", "processed", o$task)
output_dir <- pj("result", "comparison", o$task, paste0("all_", o$task))
mkdir(output_dir, remove_old = F)

l <- 7.5  # figure size
L <- 10   # figure size
m <- 0.5  # legend margin

dirs <- list(
    "midas_embed"  = pj("result", "comparison", o$task, "midas_embed", o$experiment, o$model, o$init_model),
    "multigrate"   = pj("result", "comparison", o$task, "multigrate"),
    "scmomat"      = pj("result", "comparison", o$task, "scmomat"),
    "scvaeit"      = pj("result", "comparison", o$task, "scvaeit"),
    "stabmap"      = pj("result", "comparison", o$task, "stabmap")
)

plt_c_b <- NULL
plt_c_l <- NULL

for (method in names(dirs)) {
    message(paste0("Plotting ", method))

    dir <- dirs[[method]]
    reduc <- ifelse(method == "midas_embed", "c.umap", "umap")
    obj <- LoadH5Seurat(pj(dir, "obj.h5seurat"), reductions = c(reduc))
    obj@meta.data$batch <- gsub("ica", "ICA", gsub("^BM$", "ASAP", gsub("^bm$", "CITE", obj@meta.data$batch)))
    obj@meta.data$batch <- factor(x = obj@meta.data$batch, levels = c("ICA", "ASAP", "CITE"))
    obj@meta.data$l1    <- factor(x = obj@meta.data$l1, levels = str_sort(unique(obj@meta.data$l1)))

    p1 <- dim_plot(obj, title = rename_method(method), w = L, h = L, reduction = reduc, no_axes = T, return_plt = T, display = F, border = T, raster = T,
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
w <- L * 5 + 1.3
h <- L * 2 + 1
plt_size(w, h)
ggsave(plot = plt_c, file = pj(output_dir, paste(o$experiment, o$model, o$init_model, "merged_c.png", sep = "_")), width = w, height = h, scale = 0.5, limitsize = F)
ggsave(plot = plt_c, file = pj(output_dir, paste(o$experiment, o$model, o$init_model, "merged_c.pdf", sep = "_")), width = w, height = h, scale = 0.5, limitsize = F)
