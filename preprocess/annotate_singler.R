source("/root/workspace/code/midas/preprocess/utils.R")
library(celldex)
library(SingleR)

parser <- ArgumentParser()
parser$add_argument("--task", type = "character", default = "dogma_full")
o <- parser$parse_args()
task <- o$task

config <- parseTOML("configs/data.toml")[[task]]
combs <- config$combs
comb_ratios <- config$comb_ratios
mod <- "rna"

input_dirs <- pj(config$raw_data_dirs, "seurat")
output_dirs <- pj(config$raw_data_dirs, "label_singler")
mkdirs(output_dirs, remove_old = T)

sc_list <- list()
label_main_list <- list()
label_fine_list <- list()
# hpca <- HumanPrimaryCellAtlasData()
hpca <- MonacoImmuneData()
subset_id <- "0"
for (dataset_id in seq_along(input_dirs)) {
    comb <- combs[[dataset_id]]
    comb_ratio <- comb_ratios[[dataset_id]]
    fp <- pj(input_dirs[dataset_id], paste0(mod, ".h5seurat"))
    if (file.exists(fp)) {
        prt("Loading ", fp, " ...")
        sc <- LoadH5Seurat(fp)
        cell_num <- dim(sc)[2]
        end_ids <- round(cumsum(comb_ratio) / sum(comb_ratio) * cell_num)
        start_ids <- c(1, end_ids + 1)
        for (split_id in seq_along(comb)) {
            if (mod %in% comb[[split_id]]) {
                cell_names <- colnames(sc)[start_ids[split_id]:end_ids[split_id]]
                sc_list[[subset_id]] <- subset(sc, cells = cell_names)
                sc_list[[subset_id]]$subset_id <- subset_id

                # Annotate each subset
                prt("Annotating subset ", subset_id, " ...")
                sce <- as.SingleCellExperiment(sc_list[[subset_id]])
                label_main_list[[subset_id]] <- SingleR(test = sce, ref = hpca, labels = hpca$label.main)
                label_fine_list[[subset_id]] <- SingleR(test = sce, ref = hpca, labels = hpca$label.fine)
                # table(label_main_list[[subset_id]]$labels)
                # table(label_fine_list[[subset_id]]$labels)

                # # Annotation diagnostics
                # plt <- plotScoreHeatmap(label_main_list[[subset_id]])
                # ggsave(plt, file="plotDeltaDistribution.png", width=6, height=6)

                # plt <- plotDeltaDistribution(label_main_list[[subset_id]], ncol = 5)
                # ggsave(plt, file="plotDeltaDistribution.png", width=10, height=4)

                # summary(is.na(label_main_list[[subset_id]]$pruned.labels))

                # Save labels
                write.csv(label_main_list[[subset_id]]$labels, file = pj(output_dirs[dataset_id], "main.csv"))
                write.csv(label_fine_list[[subset_id]]$labels, file = pj(output_dirs[dataset_id], "fine.csv"))
            }
            subset_id <- toString(strtoi(subset_id) + 1)
        }
    }
    else {
        subset_id <- toString(strtoi(subset_id) + length(comb))
    }
}