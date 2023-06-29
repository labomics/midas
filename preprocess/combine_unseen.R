source("/root/workspace/code/midas/preprocess/utils.R")


parser <- ArgumentParser()
parser$add_argument("--task", type = "character", default = "dogma_full_transfer")
parser$add_argument("--reference", type = "character", default = "atlas_no_dogma")
o <- parser$parse_args()
task <- o$task

config <- parseTOML("configs/data.toml")[[gsub("_vd.*|_vt.*|_transfer$|_ref_.*$", "", task)]]
combs <- config$combs
comb_ratios <- config$comb_ratios
mods_ <- unique(unlist(combs))
mods <- vector()
for (mod in c("atac", "rna", "adt")) {
    if (mod %in% mods_) {
        mods <- c(mods, mod)
    }
}

input_dirs <- pj(config$raw_data_dirs, "seurat")
task_dir <- pj("data", "processed", task)
mkdir(task_dir, remove_old = T)
output_fig_dir <- pj(task_dir, "fig")
mkdir(output_fig_dir, remove_old = T)
output_feat_dir <- pj(task_dir, "feat")
mkdir(output_feat_dir, remove_old = T)
ref_feat_dir <- pj("data", "processed", o$reference, "feat")

for (dataset_id in seq_along(input_dirs)) {
    subset_id <- toString(dataset_id - 1)
    output_dir <- pj(task_dir, paste0("subset_", subset_id))
    mkdir(pj(output_dir, "mat"), remove_old = T)
    mkdir(pj(output_dir, "mask"), remove_old = T)
}

if (grepl("_vd", task)) { # varying sequencing depths
    library(scuttle)
    if (grepl("_vd02", task)) {
        depth_factors <- c(0.02, 0.02, 1, 1)
    } else if (grepl("_vd05", task)) {
        depth_factors <- c(0.05, 0.05, 1, 1)
    } else if (grepl("_vd1", task)) {
        depth_factors <- c(0.1, 0.1, 1, 1)
    } else if (grepl("_vd2", task)) {
        depth_factors <- c(0.2, 0.2, 1, 1)
    } else if (grepl("_vd4", task)) {
        depth_factors <- c(0.4, 0.4, 1, 1)
    } else if (grepl("_vd5", task)) {
        depth_factors <- c(0.5, 0.5, 1, 1)
    } else if (grepl("_vd6", task)) {
        depth_factors <- c(0.6, 0.6, 1, 1)
    } else if (grepl("_vd8", task)) {
        depth_factors <- c(0.8, 0.8, 1, 1)
    } else {
        stop(paste0(task, ": Invalid task"))
    }
}

if (grepl("_vt", task)) { # varying cell types
    if (grepl("_vt1", task)) {
        missing_types <- c("other T", "B", "CD4 T", "CD8 T")
    } else if (grepl("_vt2", task)) {
        missing_types <- c("other T", "B", "CD8 T", "CD4 T")
    } else {
        missing_types <- c("other T", "CD8 T", "NK", "B")
    }
    label_dirs <- pj(config$raw_data_dirs, "label_seurat")
    cell_mask_list <- list()
    fn <- paste0("l1_", tail(strsplit(task, split = "_")[[1]], 1), ".csv")
    for (i in seq_along(label_dirs)) {
        l1 <- read.csv(pj(label_dirs[i], "l1.csv"), header = T)[, 2]
        mask <- l1 != missing_types[i]
        write.csv(l1[mask], file = pj(label_dirs[i], fn))
        cell_mask_list[[toString(i - 1)]] <- mask
    }
}

merge_counts <- function(mod) {

    # Load different subsets
    prt("Processing ", mod, " data ...")
    sc_list <- list()
    feat_list <- list()
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
                    feat_list[[subset_id]] <- rownames(sc_list[[subset_id]])
                }
                subset_id <- toString(strtoi(subset_id) + 1)
            }
        }
        else {
            subset_id <- toString(strtoi(subset_id) + length(comb))
        }
    }


    # Select features and get feature masks for each subset
    mask_list <- list()
    ref_feat <- read.csv(pj(ref_feat_dir, paste0("feat_names_", mod, ".csv")), header = T)[, 2]
    for (subset_id in names(sc_list)) {
        counts <- sc_list[[subset_id]][[mod]]@counts
        counts_expanded <- Matrix(nrow = length(ref_feat), ncol = dim(counts)[2],
                                  dimnames = list(ref_feat, colnames(counts)),
                                  data = 0, sparse = TRUE)
        feat_intersect <- intersect(feat_list[[subset_id]], ref_feat)
        mask_list[[subset_id]] <- as.integer(ref_feat %in% feat_intersect)
        counts_expanded[feat_intersect, ] <- counts[feat_intersect, ]
        sc_list[[subset_id]] <- CreateSeuratObject(counts = counts_expanded, assay = mod)
        sc_list[[subset_id]]$subset_id <- subset_id
        sc_list[[subset_id]] <- RenameCells(sc_list[[subset_id]], add.cell.id = paste0("U", subset_id))
    }
    feat_dims[[mod]] <<- length(ref_feat)
    write.csv(ref_feat, file = pj(output_feat_dir, paste0("feat_names_", mod, ".csv")))


    # Save
    for (subset_id in names(sc_list)) {
        prt("Saving subset ", subset_id, " ...")
        output_dir <- pj(task_dir, paste0("subset_", subset_id))
        output_mat_dir <- pj(output_dir, "mat")

        mat <- t(data.matrix(sc_list[[subset_id]][[mod]]@counts))  # N * D
        if (grepl("_vd", task)) {
            mat <- as.matrix(downsampleMatrix(mat, prop = depth_factors[strtoi(subset_id) + 1], bycol=F))
        }
        if (grepl("_vt", task)) {
            mat <- mat[cell_mask_list[[subset_id]], ]
        }
        # Save count data
        write.csv(mat, file = pj(output_mat_dir, paste0(mod, ".csv")))
        # Save cell IDs
        write.csv(rownames(mat), file = pj(output_dir, "cell_names.csv"))

        output_mask_dir <- pj(output_dir, "mask")
        mask <- t(data.matrix(mask_list[[subset_id]]))  # 1 * D
        # Save the feature mask
        write.csv(mask, file = pj(output_mask_dir, paste0(mod, ".csv")))
    }
}



merge_frags <- function() {
    mod <- "atac"
    # Load different subsets
    prt("Processing ", mod, " data ...")
    sc_list <- list()
    feat_list <- list()

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
                    feat_list[[subset_id]] <- StringToGRanges(rownames(sc_list[[subset_id]]))
                }
                subset_id <- toString(strtoi(subset_id) + 1)
            }
        }
        else {
            subset_id <- toString(strtoi(subset_id) + length(comb))
        }
    }


    # Re-compute peak counts based on reference features
    ref_feat <- read.csv(pj(ref_feat_dir, paste0("feat_names_", mod, ".csv")),
                         header = T)[, 2]
    ref_feat_gr <- StringToGRanges(ref_feat)
    for (subset_id in names(sc_list)) {
        dataset_id <- sum(strtoi(subset_id) >= c(0, cumsum(lengths(combs))))
        frag_path <- pj(config$raw_data_dirs[dataset_id],
                        config$raw_data_frags[dataset_id])
        cell_names <- colnames(sc_list[[subset_id]])
        cell_names_copy <- NULL
        if (grepl("tea", frag_path)) {
            cell_names_copy <- cell_names
            metadata <- read.csv(file = gsub("fragments.tsv", "metadata.csv", frag_path))
            cell_names <- metadata$barcodes[match(cell_names, metadata$original_barcodes)]
        }
        frags <- CreateFragmentObject(path = frag_path, cells = cell_names)
        counts <- FeatureMatrix(fragments = frags, features = ref_feat_gr, cells = cell_names)
        assay <- CreateChromatinAssay(counts = counts, fragments = frags)
        sc_list[[subset_id]] <- CreateSeuratObject(counts = assay, assay = "atac")
        sc_list[[subset_id]]$subset_id <- subset_id
        sc_list[[subset_id]] <- RenameCells(sc_list[[subset_id]], add.cell.id = paste0("U", subset_id))
        if (grepl("tea", frag_path)) {
            sc_list[[subset_id]] <- RenameCells(sc_list[[subset_id]], new.names = cell_names_copy)
        }
    }
    feat_dims[[mod]] <<- width(ref_feat_gr@seqnames)
    write.csv(ref_feat, file = pj(output_feat_dir, paste0("feat_names_", mod, ".csv")))


    # Save
    for (subset_id in names(sc_list)) {
        prt("Saving subset ", subset_id, " ...")
        output_dir <- pj(task_dir, paste0("subset_", subset_id))
        output_mat_dir <- pj(output_dir, "mat")

        mat <- t(data.matrix(sc_list[[subset_id]][[mod]]@counts))  # N * D
        if (grepl("_vd", task)) {
            mat <- as.matrix(downsampleMatrix(mat, prop = depth_factors[strtoi(subset_id) + 1], bycol=F))
        }
        if (grepl("_vt", task)) {
            mat <- mat[cell_mask_list[[subset_id]], ]
        }
        # Save count data
        write.csv(mat, file = pj(output_mat_dir, paste0(mod, ".csv")))
        # Save cell IDs
        write.csv(rownames(mat), file = pj(output_dir, "cell_names.csv"))
    }
}


feat_dims <- list()
for (mod in mods) {
    if (mod == "atac") {
        merge_frags()
    } else {
        merge_counts(mod)
    }
}
# Save feature dimensionalities
prt("feat_dims: ", feat_dims)
write.csv(feat_dims, file = pj(output_feat_dir, "feat_dims.csv"))