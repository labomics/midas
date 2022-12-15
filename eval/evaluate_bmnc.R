# Evaluate imputation and classification

library(Seurat)
library(cowplot)
library(dplyr)
library(ggplot2)
library(ggpol)
library(bmcite.SeuratData)
library(stringr)
library(mclust)
library(assert)
library(ggpubr)
library(caret)
library(argparse)


parser <- ArgumentParser()
parser$add_argument("--task", type="character", default="hbm")
parser$add_argument("--exp", type="character", default="sup")
parser$add_argument("--model", type="character", default="default")
o <- parser$parse_args()



# Initialize
data_dir <- "data/hbm/preprocessed"
data_mat_dir <- file.path(data_dir, "mat")
data_name_dir <- file.path(data_dir, "name")
data_vec_dir <- file.path(data_dir, "vec")



cell_names <- read.csv(file.path(data_name_dir, "cell.csv"), header=T)[, 2]
rna_names <- read.csv(file.path(data_name_dir, "rna.csv"), header=T)[, 2]
adt_names <- read.csv(file.path(data_name_dir, "adt.csv"), header=T)[, 2]
cell_num <- length(cell_names)


result_dir <- file.path("result", o$task, o$exp, o$model, "transform")
cat(paste("\n\n========================\nParsing data from ", result_dir, "\n"))


load_mat <- function(directory, mod, file_pattern) {
    fnames <- dir(path = directory, pattern = file_pattern)
    fnames <- str_sort(fnames, decreasing = FALSE)
    values <- list()
    for (n in seq_len(length(fnames))) {
        values[[n]] <- read.csv(file.path(directory, fnames[n]), header = FALSE)
    }
    mat <- data.matrix(bind_rows(values))
    rownames(mat) <- cell_names[(cell_num - dim(mat)[1] + 1):cell_num]
    if (mod == "rna") {
        colnames(mat) <- rna_names
    } else if (mod == "adt") {
        colnames(mat) <- adt_names
    } else if (mod == "label") {
        colnames(mat) <- "class"
    } else {
        stop(paste0(mod, ": Invalid modality"))
    }
    return(mat)
}



plot_fig <- function(input_mods, pred_mod, fig_type) {
    input_mods_str <- paste0("from_['", paste(input_mods, collapse = "', '"), "']")
    input_mods_dir <- file.path(result_dir, input_mods_str)
    pred_mods_dir <- file.path(input_mods_dir, pred_mod)
    fig_dir <- file.path(pred_mods_dir, "fig", fig_type)
    dir.create(fig_dir, recursive=T, showWarnings=F)

    pred_mat <- load_mat(pred_mods_dir, pred_mod, "*_pred.csv")
    gt_mat <- load_mat(pred_mods_dir, pred_mod, "*_gt.csv")

    if (pred_mod == "adt") {
        pear_cors <- list()
        for (protn_name in adt_names) {
            gt_vec <- gt_mat[, protn_name]
            pred_vec <- pred_mat[, protn_name]
            # plt <- qplot(gt_vec, pred_vec) + geom_smooth(method="lm") +
            #     geom_point(colour = "black", size = 0.3) +
            #     labs(x=paste(protn_name, "gt", sep="_"), y=paste(protn_name, "pred", sep="_")) +
            #     stat_cor(method = "pearson", digits=3, label.y=max(pred_vec)*1.1)
            # ggsave(file = file.path(fig_dir, paste(protn_name, "png", sep=".")), plot=plt)
            pear_cor <- cor(gt_vec, pred_vec, method="pearson")
            # print(paste(protn_name, "pearson correlation:", pear_cor), quote=F)
            pear_cors[protn_name] <- pear_cor
        }
        cat(paste0("from ", paste(input_mods, collapse=" + "), ":\n"))
        # print(t(pear_cors))
        mean_cor <- mean(unlist(pear_cors))
        median_cor <- median(unlist(pear_cors))
        cat(paste("    averaged Pearson correlation:",
            round(mean_cor, 4), "(mean), ", round(median_cor, 4), "(median), ", "\n"))
    }

    if (pred_mod == "label") {
        gt <- gt_mat[, "class"]
        pred <- pred_mat[, "class"]
        plt <- ggplot() + 
            geom_confmat(aes(x = gt, y = pred), normalize=T, text.perc=T, text.digits=2,
                text.size=2) +
            scale_x_continuous("annotated type", breaks=c(0:max(gt)), minor_breaks=c(0:max(gt))) +
            scale_y_continuous("predicted type", breaks=c(0:max(pred)), minor_breaks=c(0:max(pred)))
        ggsave(file = file.path(fig_dir, paste("confusion", "png", sep=".")), plot=plt,
            width=12, height=6)
        cm <- confusionMatrix(factor(pred), factor(gt), mode="prec_recall")
        cat(paste0("from ", paste(input_mods, collapse=" + "), ":\n"))
        # print(cm$overall)
        f1 <- mean(cm$byClass[, "F1"])
        cat(paste("    averaged F1 score:", round(f1, 4), "\n"))
    }
}


# Predict ADT
fig_type <- "scatter"
cat("\nPearson correlation:\n")
plot_fig(input_mods=c("rna"), pred_mod="adt", fig_type=fig_type)
# plot_fig(input_mods=c("adt"), pred_mod="adt", fig_type=fig_type)
# plot_fig(input_mods=c("label"), pred_mod="adt", fig_type=fig_type)
# plot_fig(input_mods=c("rna", "adt"), pred_mod="adt", fig_type=fig_type)
# plot_fig(input_mods=c("rna", "label"), pred_mod="adt", fig_type=fig_type)
# plot_fig(input_mods=c("rna", "adt", "label"), pred_mod="adt", fig_type=fig_type)


# Predict cell type label
fig_type <- "confusion"
cat("\nF1 score:\n")
plot_fig(input_mods=c("rna"), pred_mod="label", fig_type=fig_type)
plot_fig(input_mods=c("adt"), pred_mod="label", fig_type=fig_type)
plot_fig(input_mods=c("rna", "adt"), pred_mod="label", fig_type=fig_type)