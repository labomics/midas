setwd("/root/workspace/code/midas/eval/plot")
source("/root/workspace/code/midas/utils/utils.R")
library(tibble)
library(RColorBrewer)
library(dynutils)
library(stringr)
library(Hmisc)
library(plyr)
library(gdata)
source("knit_table.R")

parser <- ArgumentParser()
parser$add_argument("--fig", type = "integer", default = 16)
o <- parser$parse_known_args()[[1]]

method <- "midas"
if (o$fig == 1) {
    # dogma scib (main)
    dataset <- "dogma"
    sota <- T
    mosaic <- F
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/fig 2"
    img_head <- "Fig2c_"
}
if (o$fig == 2) {
    # teadog scib
    dataset <- "teadog"
    sota <- T
    mosaic <- F
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 2"
    img_head <- "sFig2_"
}
if (o$fig == 3) {
    # dogma scmib (main)
    dataset <- "dogma"
    sota <- F
    mosaic <- T
    scmib <- T
    transfer <- F
    outdir <- "../../paper/metrics/fig 3"
    img_head <- "Fig3b_"
}
if (o$fig == 4) {
    # teadog scmib
    dataset <- "teadog"
    sota <- F
    mosaic <- T
    scmib <- T
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 3"
    img_head <- "sFig3_"
}
if (o$fig == 5) {
    # dogma + sota, scib
    dataset <- "dogma"
    sota <- T
    mosaic <- T
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 3"
    img_head <- "sFig3_"
}
if (o$fig == 6) {
    # teadog + sota, scib
    dataset <- "teadog"
    sota <- T
    mosaic <- T
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 3"
    img_head <- "sFig3_"
}
if (o$fig == 7) {
    # dogma transfer, scmib
    dataset <- "dogma"
    sota <- F
    mosaic <- T
    scmib <- T
    transfer <- T
    outdir <- "../../paper/metrics/supplementary fig 5"
    img_head <- "sFig5_"
}
if (o$fig == 8) {
    # dogma transfer + sota, scib
    dataset <- "dogma"
    sota <- T
    mosaic <- T
    scmib <- F
    transfer <- T
    outdir <- "../../paper/metrics/supplementary fig 5"
    img_head <- "sFig5_"
}



# comparing mosaic methods
if (o$fig == 9) {
    # dogma mosaic, scib
    method <- "scmomat"
    dataset <- "dogma"
    sota <- F
    mosaic <- T
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 3"
    img_head <- paste0("sFig3_", method, "_")
}
if (o$fig == 10) {
    # teadog mosaic, scib
    method <- "scmomat"
    dataset <- "teadog"
    sota <- F
    mosaic <- T
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 3"
    img_head <- paste0("sFig3_", method, "_")
}
if (o$fig == 11) {
    # dogma mosaic, scib
    method <- "stabmap"
    dataset <- "dogma"
    sota <- F
    mosaic <- T
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 3"
    img_head <- paste0("sFig3_", method, "_")
}
if (o$fig == 12) {
    # teadog mosaic, scib
    method <- "stabmap"
    dataset <- "teadog"
    sota <- F
    mosaic <- T
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 3"
    img_head <- paste0("sFig3_", method, "_")
}
if (o$fig == 13) {
    # dogma mosaic, scib
    method <- "scvaeit"
    dataset <- "dogma"
    sota <- F
    mosaic <- T
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 3"
    img_head <- paste0("sFig3_", method, "_")
}
if (o$fig == 14) {
    # teadog mosaic, scib
    method <- "scvaeit"
    dataset <- "teadog"
    sota <- F
    mosaic <- T
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 3"
    img_head <- paste0("sFig3_", method, "_")
}
if (o$fig == 15) {
    # dogma mosaic, scib
    method <- "multigrate"
    dataset <- "dogma"
    sota <- F
    mosaic <- T
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 3"
    img_head <- paste0("sFig3_", method, "_")
}
if (o$fig == 16) {
    # teadog mosaic, scib
    method <- "multigrate"
    dataset <- "teadog"
    sota <- F
    mosaic <- T
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 3"
    img_head <- paste0("sFig3_", method, "_")
}
if (o$fig == 17) {
    # dogma mosaic, scib
    method <- "midas"
    dataset <- "dogma"
    sota <- F
    mosaic <- T
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 3"
    img_head <- paste0("sFig3_", method, "_")
}
if (o$fig == 18) {
    # teadog mosaic, scib
    method <- "midas"
    dataset <- "teadog"
    sota <- F
    mosaic <- T
    scmib <- F
    transfer <- F
    outdir <- "../../paper/metrics/supplementary fig 3"
    img_head <- paste0("sFig3_", method, "_")
}

# de novo, dogma single
method <- "midas"
dataset <- "dogma"
sota <- F
mosaic <- T
scmib <- F
transfer <- F
outdir <- "../../paper/metrics/supplementary fig 3"
img_head <- "sFig3_"

xls_metrics_path <- paste0("data/scib_metrics_sota+mosaic_dogma_single_atac_e0_l_2_sp_00001899_sorted_R1_12.xlsx")
xls_metrics_path <- paste0("data/scib_metrics_sota+mosaic_dogma_single_rna_e0_l_2_sp_00001999_sorted_R1_12.xlsx")
# xls_metrics_path <- paste0("data/scib_metrics_sota+mosaic_dogma_single_adt_e0_default_sp_00001899_sorted_R1_12.xlsx")


# de novo, bm
method <- "midas"
dataset <- "bm"
scmib <- F
transfer <- F
outdir <- "../../paper/metrics/supplementary fig 6"
img_head <- "sFig6_"

xls_metrics_path <- paste0("data/scib_metrics_sota+mosaic_bm_e0_la_1_sp_00001899_sorted.xlsx")


# transfer, bm
method <- "midas"
dataset <- "bm"
scmib <- F
transfer <- F
outdir <- "../../paper/metrics/supplementary fig 6"
img_head <- "sFig6_"

xls_metrics_path <- paste0("data/scib_metrics_mosaic_bm_no_map_ref_default_sp_00003599_sorted.xlsx")

# init <- ifelse(transfer, "sp_00003699", "sp_00001899")

# if (sota & !mosaic) {
#     xls_metrics_path <- paste0("data/scib_metrics_sota_",        dataset, "_e0_", init, "_sorted.xlsx")
# } else if (!sota & mosaic) {
#     if (method == "midas") {
#         xls_metrics_path <- paste0("data/scib_metrics_mosaic_",      dataset, "_e0_", init, "_sorted.xlsx")
#     } else {
#         xls_metrics_path <- paste0("data/scib_metrics_mosaic_", dataset, "_", method, "_sorted.xlsx")
#     }
# } else {
#     if (dataset == "dogma" & !transfer) {
#          xls_metrics_path <- paste0("data/scib_metrics_sota+mosaic_", dataset, "_e0_", init, "_sorted+less_mod.xlsx")
#     } else {
#         xls_metrics_path <- paste0("data/scib_metrics_sota+mosaic_", dataset, "_e0_", init, "_sorted.xlsx")
#     }
# }

# if (scmib) {
#     xls_metrics_path <- paste0("data/scmib_metrics_", dataset, "_", init, "_sorted.xlsx")
# }

# mkdir(outdir, remove_old = F)

metrics_tab <- read.xls(xls_metrics_path)
# metrics_tab <- as.data.frame(metrics_tab[, -1])
metrics_tab <- as.data.frame(metrics_tab)
names(metrics_tab)[1] <- "Method"
metrics_tab <- metrics_tab[order(metrics_tab$`overall_score`,  decreasing = T), ]


metrics_tab <- metrics_tab[metrics_tab[["Method"]] != "midas_feat+wnn", ]

if (scmib) {
    group_batch <- c("iLISI_feat",  "graph_conn_feat",  "kBET_feat", "iLISI_embed", "graph_conn_embed", "kBET_embed")
    group_mod <- c("ASW_mod", "FOSCTTM", "Label_transfer", "AUROC", "Pearson_RNA", "Pearson_ADT")
    group_bio <- c("NMI_feat",  "ARI_feat",  "il_score_f1_feat",  "cLISI_feat", "NMI_embed", "ARI_embed", "il_score_f1_embed", "cLISI_embed")
    metrics_tab <- add_column(metrics_tab, "Task" = metrics_tab[, 1], .after = "Method")
    metrics_tab[["Method"]] <- "midas_embed"
} else {
    group_batch <- c("iLISI", "graph_conn", "kBET")
    group_mod <- NULL
    group_bio <- c("NMI", "ARI", "il_score_f1", "cLISI")
}
n_metrics_batch <- length(group_batch)
n_metrics_mod <- length(group_mod)
n_metrics_bio <- length(group_bio)

names(metrics_tab) <- str_replace_all(names(metrics_tab), c(
    "iLISI" = "Graph iLISI",
    "graph_conn" = "Graph connectivity",
    "il_score_f1" = "Isolated label F1",
    "batch_score" = "Batch correction",
    "ASW_mod" = "Modality ASW",
    "Label_transfer" = "Label transfer F1",
    "AUROC" = "ATAC AUROC",
    "Pearson_RNA" = "RNA Pearson's R",
    "Pearson_ADT" = "ADT Pearson's R",
    "mod_score" = "Modality alignment",
    "cLISI" = "Graph cLISI",
    "bio_score" = "Biological conservation",
    "overall_score" = "Overall score",
    "_embed" = " (embed)",
    "_feat" = " (feat)"
))
metrics_tab[, "Method"] <- str_replace_all(metrics_tab[, "Method"], c(
    "midas_embed" = "MIDAS",
    "midas_feat\\+wnn" = "MIDAS-feat + WNN",
    "bbknn" = "BBKNN + average",
    "harmony\\+wnn" = "Harmony + WNN",
    "seurat_cca\\+wnn" = "Seurat-CCA + WNN",
    "seurat_rpca\\+wnn" = "Seurat-RPCA + WNN",
    "scanorama_embed\\+wnn" = "Scanorama-embed + WNN",
    "pca\\+wnn" = "PCA + WNN",
    "scanorama_feat\\+wnn" = "Scanorama-feat + WNN",
    "liger\\+wnn" = "LIGER + WNN",
    "mofa" = "MOFA+",
    "multigrate" = "Multigrate",
    "stabmap" = "StabMap",
    "scvaeit" = "scVAEIT",
    "scmomat" = "scMoMaT"
))
metrics_tab[, "Task"] <- str_replace_all(metrics_tab[, "Task"], c(
    "dogma_|teadog_|_transfer" = ""
    # "_" = "-",
    # "paired-full" = "paired+full",
    # "single-full" = "single+full",
    # "-single$" = "-single-diag"
))
if (transfer) {
    metrics_tab[, "Method"] <- str_replace_all(metrics_tab[, "Method"], c("MIDAS" = "MIDAS (transfer)"))
}

if (dataset == "bm") {
    metrics_tab[, "Task"] <- "full"
    dataset <- "dogma"
}

if (scmib) {
    column_info <- data.frame(id = colnames(metrics_tab),
        group = c("Text", "Image", rep("batch_score", (1 + n_metrics_batch)),
                                rep("mod_score", (1 + n_metrics_mod)),
                                rep("bio_score", (1 + n_metrics_bio)), "overall_score"),
        geom = c("text", "image", rep("circle", n_metrics_batch), "bar",
                                rep("circle", n_metrics_mod), "bar",
                                rep("circle", n_metrics_bio), "bar", "bar"),
        width = c(   10.5,   2.5, rep(1, n_metrics_batch), 2,
                                rep(1, n_metrics_mod), 2,
                                rep(1, n_metrics_bio), 2, 2),
        overlay = F)
} else {
    column_info <- data.frame(id = colnames(metrics_tab),
                            group = c("Text", "Image", rep("batch_score", (1 + n_metrics_batch)),
                                                        rep("bio_score", (1 + n_metrics_bio)), "overall_score"),
                            geom =  c("text", "image", rep("circle", n_metrics_batch), "bar",
                                                        rep("circle", n_metrics_bio), "bar", "bar"),
                            width = c(  12,     2.5, rep(1, n_metrics_batch), 2,
                                                        rep(1, n_metrics_bio), 2, 2),
                            overlay = F)
}


# https://www.datanovia.com/en/blog/top-r-color-palettes-to-know-for-great-data-visualization/
palettes <- list("overall_score" = "Oranges",
                 "batch_score" = "Greens",
                 "mod_score" = "PuRd",
                 "bio_score" = "Blues")

g <- scIB_knit_table(data = metrics_tab, dataset = dataset, column_info = column_info, row_info = data.frame(id = metrics_tab$Method),
                     palettes = palettes, task = T, usability = F)

if (scmib) {
    w <- 300
    h <- 150
} else {
    w <- 250
    h <- 230
}
options(repr.plot.width = w / 25.4, repr.plot.height = h / 25.4)
g

now <- Sys.time()
ggsave(paste0(outdir, "/", img_head, strsplit(basename(xls_metrics_path), split = "\\.")[[1]][1],
    # format(now, "_%Y%m%d_%H%M%S"),  ".png"), g, device = "png", dpi = "retina",
    ".png"), g, device = "png", dpi = "retina",
    width = w, height = h, units = "mm")
ggsave(paste0(outdir, "/", img_head, strsplit(basename(xls_metrics_path), split = "\\.")[[1]][1],
    ".pdf"), g, device = "pdf", dpi = "retina",
    width = w, height = h, units = "mm")
