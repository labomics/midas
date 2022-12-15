source("/root/workspace/code/sc-transformer/preprocess/utils.R")
setwd("/root/workspace/code/sc-transformer/")
library(RColorBrewer)
library(patchwork)

parser <- ArgumentParser()
parser$add_argument("--task", type = "character", default = "dogma")
parser$add_argument("--method", type = "character", default = "midas_embed")
parser$add_argument("--experiment", type = "character", default = "e0")
parser$add_argument("--model", type = "character", default = "default")
parser$add_argument("--init_model", type = "character", default = "sp_00001899")
parser$add_argument("--init_model_transfer", type = "character", default = "sp_00003699")
o <- parser$parse_known_args()[[1]]

output_dir <- pj("paper", "5", "a")
mkdir(output_dir, remove_old = F)

l <- 7.5  # figure size
L <- 10   # figure size
m <- 0.5  # legend margin


if (o$task == "dogma") {
    dirs <- list(
        "dogma_full"           = pj("result", "comparison", "dogma_full"       , o$method, o$experiment, o$init_model),
        "dogma_paired_full"    = pj("result", "comparison", "dogma_paired_full", o$method, o$experiment, o$init_model),
        "dogma_paired_abc"     = pj("result", "comparison", "dogma_paired_abc" , o$method, o$experiment, o$init_model),
        "dogma_paired_ab"      = pj("result", "comparison", "dogma_paired_ab"  , o$method, o$experiment, o$init_model),
        "dogma_paired_ac"      = pj("result", "comparison", "dogma_paired_ac"  , o$method, o$experiment, o$init_model),
        "dogma_paired_bc"      = pj("result", "comparison", "dogma_paired_bc"  , o$method, o$experiment, o$init_model),
        "dogma_single_full"    = pj("result", "comparison", "dogma_single_full", o$method, o$experiment, o$init_model),
        "dogma_single"         = pj("result", "comparison", "dogma_single"     , o$method, o$experiment, o$init_model),
        "dogma_paired_a"       = pj("result", "comparison", "dogma_paired_a"   , o$method, o$experiment, o$init_model),
        "dogma_paired_b"       = pj("result", "comparison", "dogma_paired_b"   , o$method, o$experiment, o$init_model),
        "dogma_paired_c"       = pj("result", "comparison", "dogma_paired_c"   , o$method, o$experiment, o$init_model),
        "dogma_single_atac"    = pj("result", "comparison", "dogma_single_atac", o$method, o$experiment, o$init_model),
        "dogma_single_rna"     = pj("result", "comparison", "dogma_single_rna" , o$method, o$experiment, o$init_model),
        "dogma_single_adt"     = pj("result", "comparison", "dogma_single_adt" , o$method, o$experiment, o$init_model)
    )
} else {
    dirs <- list(
        "dogma_full_transfer"           = pj("result", "comparison", "dogma_full_transfer"       , o$method, o$experiment, o$init_model_transfer),
        "dogma_paired_full_transfer"    = pj("result", "comparison", "dogma_paired_full_transfer", o$method, o$experiment, o$init_model_transfer),
        "dogma_paired_abc_transfer"     = pj("result", "comparison", "dogma_paired_abc_transfer" , o$method, o$experiment, o$init_model_transfer),
        "dogma_paired_ab_transfer"      = pj("result", "comparison", "dogma_paired_ab_transfer"  , o$method, o$experiment, o$init_model_transfer),
        "dogma_paired_ac_transfer"      = pj("result", "comparison", "dogma_paired_ac_transfer"  , o$method, o$experiment, o$init_model_transfer),
        "dogma_paired_bc_transfer"      = pj("result", "comparison", "dogma_paired_bc_transfer"  , o$method, o$experiment, o$init_model_transfer),
        "dogma_single_full_transfer"    = pj("result", "comparison", "dogma_single_full_transfer", o$method, o$experiment, o$init_model_transfer),
        "dogma_single_transfer"         = pj("result", "comparison", "dogma_single_transfer"     , o$method, o$experiment, o$init_model_transfer),
        "dogma_paired_a_transfer"       = pj("result", "comparison", "dogma_paired_a_transfer"   , o$method, o$experiment, o$init_model_transfer),
        "dogma_paired_b_transfer"       = pj("result", "comparison", "dogma_paired_b_transfer"   , o$method, o$experiment, o$init_model_transfer),
        "dogma_paired_c_transfer"       = pj("result", "comparison", "dogma_paired_c_transfer"   , o$method, o$experiment, o$init_model_transfer),
        "dogma_single_atac_transfer"    = pj("result", "comparison", "dogma_single_atac_transfer", o$method, o$experiment, o$init_model_transfer),
        "dogma_single_rna_transfer"     = pj("result", "comparison", "dogma_single_rna_transfer" , o$method, o$experiment, o$init_model_transfer),
        "dogma_single_adt_transfer"     = pj("result", "comparison", "dogma_single_adt_transfer" , o$method, o$experiment, o$init_model_transfer)
    )
}


source("/root/workspace/code/sc-transformer/preprocess/utils.R")
plt_c_b <- NULL
plt_c_l <- NULL

for (task in names(dirs)) {
    message(paste0("Plotting ", task))
    dir <- dirs[[task]]

    obj <- LoadH5Seurat(pj(dir, "obj.h5seurat"), reductions = c("c.umap"))

    p1 <- dim_plot(obj, title = task, w = L, h = L, reduction = "c.umap", no_axes = T, return_plt = T, display = F, border = T,
        split.by = NULL, group.by = "batch", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_4, legend = F)

    p2 <- dim_plot(obj, w = L, h = L, reduction = "c.umap", no_axes = T, return_plt = T, display = F, border = T,
        split.by = NULL, group.by = "l1", label = F, repel = T, label.size = 4, pt.size = 0.1, cols = col_8, legend = F)


    p1 <- p1 + labs(colour = "Sample")
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
w <- L * 14 + 1.38 * 2
h <- L * 2
plt_size(w, h)
fname <- ifelse(o$task == "dogma", "c_merged.png", "c_merged_transfer.png")
ggsave(plot = plt_c, file = pj(output_dir, fname), width = w, height = h, scale = 0.4, limitsize = F)

