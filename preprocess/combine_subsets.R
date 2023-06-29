source("/root/workspace/code/midas/preprocess/utils.R")


parser <- ArgumentParser()
parser$add_argument("--task", type = "character", default = "atlas")
o <- parser$parse_args()
# o <- parser$parse_known_args()[[1]]  # for python interactive
task <- o$task

config <- parseTOML("configs/data.toml")[[gsub("_vd.*|_vt.*", "", task)]]
combs <- config$combs
comb_ratios <- config$comb_ratios
mods_ <- unique(unlist(combs))
mods <- vector()
for (mod in c("atac", "rna", "adt")) {
    if (mod %in% mods_) {
        mods <- c(mods, mod)
    }
}

adt_genes <- get_adt_genes()

input_dirs <- pj(config$raw_data_dirs, "seurat")
task_dir <- pj("data", "processed", task)
mkdir(task_dir, remove_old = T)
output_fig_dir <- pj(task_dir, "fig")
mkdir(output_fig_dir, remove_old = T)
output_feat_dir <- pj(task_dir, "feat")
mkdir(output_feat_dir, remove_old = T)

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
                    sc_list[[subset_id]]$subset_id <- subset_id
                    if (mod == "rna") {
                        sc_list[[subset_id]] <- remove_sparse_genes(sc_list[[subset_id]],
                                                                    kept_genes = adt_genes)
                    }
                    feat_list[[subset_id]] <- rownames(sc_list[[subset_id]])
                }
                subset_id <- toString(strtoi(subset_id) + 1)
            }
        }
        else {
            subset_id <- toString(strtoi(subset_id) + length(comb))
        }
    }
    feat_union <- Reduce(union, feat_list)

    # # If containing few subsets, just use intersected RNA features for robustness
    # if (length(sc_list) <= 5 & mod == "rna") {
    #     feat_intersect <- Reduce(intersect, feat_list)
    #     sc_list <- lapply(sc_list, subset, features = feat_intersect)
    # }

    # # debugging for adt
    # map(feat_list, length)
    # fl <- feat_list[c("0", "4", "8")]
    # map(fl, length)

    # Reduce(setdiff, feat_list[c("0", "4")])
    # Reduce(setdiff, feat_list[c("4", "0")])

    # Reduce(setdiff, fl[c("0", "8")])
    # Reduce(setdiff, fl[c("8", "0")])

    # Reduce(setdiff, fl[c("4", "8")])
    # Reduce(setdiff, fl[c("8", "4")])

    # str_sort(unlist(fl[1]))
    # str_sort(unlist(fl[2]))

    # str_extract(unlist(fl[1]), pattern = "^cd3$")
    # str_extract(unlist(fl[2]), pattern = "cd56.*")
    # str_extract(unlist(fl[3]), pattern = "^cd3$")

    # length(Reduce(intersect, feat_list[c("0", "4")]))
    # length(Reduce(union, feat_list[c("0", "4")]))

    # length(Reduce(intersect, fl[c("0", "8")]))
    # length(Reduce(union, fl[c("0", "8")]))

    # length(Reduce(intersect, fl[c("4", "8")]))
    # length(Reduce(union, fl[c("4", "8")]))


    # Remove low-frequency features
    mask_list <- list()
    mask_sum_list <- list()
    cell_num_total <- 0
    for (subset_id in names(feat_list)) {
        mask_list[[subset_id]] <- as.integer(feat_union %in% feat_list[[subset_id]])
        cell_num <- dim(sc_list[[subset_id]])[2]
        mask_sum_list[[subset_id]] <- mask_list[[subset_id]] * cell_num
        cell_num_total <- cell_num_total + cell_num
    }
    mask_sum_total <- Reduce(`+`, mask_sum_list)
    mask_ratio <- mask_sum_total / cell_num_total
    feat_union <- feat_union[mask_sum_total > 5000 | mask_ratio > 0.5]

    # Find highly variable features
    var_feat_list <- list()
    for (subset_id in names(sc_list)) {
        sc_list[[subset_id]] <- subset(sc_list[[subset_id]], features = feat_union)
        if (mod == "rna") {
            sc_list[[subset_id]] <- FindVariableFeatures(sc_list[[subset_id]], nfeatures = 5000)
        } else if (mod == "adt") {
            VariableFeatures(sc_list[[subset_id]]) <- rownames(sc_list[[subset_id]])
        } else {
            stop(paste0(mod, ": Invalid modality"))
        }
        var_feat_list[[subset_id]] <- VariableFeatures(sc_list[[subset_id]])
    }

    # Only keep features belong to the union of variable features
    var_feat_union <- Reduce(union, var_feat_list)
    if (mod == "rna") {
        var_feat_union <- union(var_feat_union, adt_genes)
    }
    sc_list <- lapply(sc_list, subset, features = var_feat_union)
    prt("Length of var_feat_union: ", length(var_feat_union))
    prt("Feature numbers of sc_list: ")
    lst <- lapply(sc_list, FUN = function(x) {length(rownames(x))})
    df <- as.data.frame(lst)
    colnames(df) <- names(lst)
    print(df)

    # Align features by merging different subsets, with missing features filled by zeros
    subset_num <- length(sc_list)
    if (subset_num > 1) {
        sc_merge <- merge(sc_list[[1]], unlist(sc_list[2:subset_num]),
            add.cell.ids = paste0("B", names(sc_list)), merge.data = T)
    } else {
        sc_merge <- RenameCells(sc_list[[1]], add.cell.id = paste0("B", names(sc_list)[1]))
    }
    feat_merged <- rownames(sc_merge)
    rownames(sc_merge[[mod]]@counts) <- feat_merged  # correct feature names for count data

    # Split into subsets and select features
    sc_split <- SplitObject(sc_merge, split.by = "subset_id")
    if (mod == "rna") {
        # Re-select 4000 variable features for each subset, rank all selected features, and keep
        # the top 4000 as the final variable features
        var_feat_integ <- SelectIntegrationFeatures(sc_split, fvf.nfeatures = 4000, nfeatures = 4000)
        var_feat_integ <- intersect(union(var_feat_integ, adt_genes), feat_merged)
    } else {
        var_feat_integ <- feat_merged
    }
    feat_dims[[mod]] <<- length(var_feat_integ)
    write.csv(var_feat_integ, file = pj(output_feat_dir, paste0("feat_names_", mod, ".csv")))
    sc_split <- lapply(sc_split, subset, features = var_feat_integ)

    # Get feature masks for each subset
    mask_list <- list()
    for (subset_id in names(sc_split)) {
        mask_list[[subset_id]] <- as.integer(var_feat_integ %in% rownames(sc_list[[subset_id]]))
    }
    prt("Feature numbers of sc_split: ")
    lst <- lapply(mask_list, sum)
    df <- as.data.frame(lst)
    colnames(df) <- names(lst)
    print(df)

    # Save subsets
    for (subset_id in names(sc_split)) {
        prt("Saving subset ", subset_id, " ...")
        output_dir <- pj(task_dir, paste0("subset_", subset_id))
        output_mat_dir <- pj(output_dir, "mat")

        mat <- t(data.matrix(sc_split[[subset_id]][[mod]]@counts))  # N * D
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
    feat_merged <- Signac::reduce(do.call("c", unname(feat_list)))

    # Filter out bad peaks based on length
    feat_widths <- width(feat_merged)
    feat_merged <- feat_merged[feat_widths < 10000 & feat_widths > 20]
    feat_merged

    # Re-compute peak counts based on merged features
    for (subset_id in names(sc_list)) {
        dataset_id <- sum(strtoi(subset_id) >= c(0, cumsum(lengths(combs))))
        frag_path <- pj(config$raw_data_dirs[dataset_id], config$raw_data_frags[dataset_id])
        cell_names <- colnames(sc_list[[subset_id]])
        cell_names_copy <- NULL
        if (grepl("tea", frag_path)) {
            cell_names_copy <- cell_names
            metadata <- read.csv(file = gsub("fragments.tsv", "metadata.csv", frag_path))
            cell_names <- metadata$barcodes[match(cell_names, metadata$original_barcodes)]
        }
        frags <- CreateFragmentObject(path = frag_path, cells = cell_names)
        counts <- FeatureMatrix(fragments = frags, features = feat_merged, cells = cell_names)
        assay <- CreateChromatinAssay(counts = counts, fragments = frags)
        sc_list[[subset_id]] <- CreateSeuratObject(counts = assay, assay = "atac")
        sc_list[[subset_id]]$subset_id <- subset_id
        if (grepl("tea", frag_path)) {
            sc_list[[subset_id]] <- RenameCells(sc_list[[subset_id]], new.names = cell_names_copy)
        }
    }

    # Remove low-frequency features for each subset
    var_feat_list <- list()
    for (subset_id in names(sc_list)) {
        # sc_list[[subset_id]] <- FindTopFeatures(sc_list[[subset_id]], min.cutoff = "q75")
        cell_num <- dim(sc_list[[subset_id]])[2]
        feat_ratio <- rowSums(sc_list[[subset_id]]$atac@counts > 0) / cell_num
        hist(feat_ratio, xlim = range(0, 1), breaks = seq(0, 1, l = 300))
        var_feat_list[[subset_id]] <- rownames(sc_list[[subset_id]])[feat_ratio > 0.04]
    }
    var_feat_union <- Reduce(union, var_feat_list)

    # Select features for each subset
    sc_list <- lapply(sc_list, subset, features = var_feat_union)

    # Merge different subsets
    subset_num <- length(sc_list)
    if (subset_num > 1) {
        sc_merge <- merge(sc_list[[1]], unlist(sc_list[2:subset_num]),
            add.cell.ids = paste0("B", names(sc_list)), merge.data = T)
    } else {
        sc_merge <- RenameCells(sc_list[[1]], add.cell.id = paste0("B", names(sc_list)[1]))
    }
    feat_merged <- rownames(sc_merge)
    rownames(sc_merge[[mod]]@counts) <- feat_merged  # correct feature names for count data
    # sort features
    gr_sorted <- sort(StringToGRanges(feat_merged))
    feat_dims[[mod]] <<- width(gr_sorted@seqnames)
    feat_sorted <- GRangesToString(gr_sorted)
    write.csv(feat_sorted, file = pj(output_feat_dir, paste0("feat_names_", mod, ".csv")))

    # Split into subsets and save
    sc_split <- SplitObject(sc_merge, split.by = "subset_id")
    for (subset_id in names(sc_split)) {
        prt("Saving subset ", subset_id, " ...")
        output_dir <- pj(task_dir, paste0("subset_", subset_id))
        output_mat_dir <- pj(output_dir, "mat")

        mat <- t(data.matrix(sc_split[[subset_id]][[mod]]@counts)[feat_sorted, ])  # N * D
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