




setwd("/data1/chenyaowen/workspace/MIDAS/stabmap/midas/")
source("preprocess/utils.R")

library(argparse)
library(RcppTOML)
library(stringr)
library(dplyr)
library(EnsDb.Hsapiens.v86)
library(purrr)



library(StabMap)
library(SingleCellMultiModal)
library(scran)
library(MultiAssayExperiment)
library(scater)
library(Signac)

#scl enable devtoolset-8 bash
#R
#install.packages("RcppTOML")

parser <- ArgumentParser()
parser$add_argument("--task", type = "character", default = "dogma_full")
parser$add_argument("--method", type = "character", default = "midas_embed")
parser$add_argument("--exp", type = "character", default = "e0")
o <- parser$parse_known_args()[[1]]

o[['method']] = 'stabmap'
#for (task in c('dogma_single')) {
for (task in c('teadog_full','teadog_paired_ab','teadog_paired_abc','teadog_paired_ac','teadog_paired_bc','teadog_paired_full','teadog_single_full','teadog_single')) {
#for (task in c('dogma_full','dogma_paired_ab','dogma_paired_abc','dogma_paired_ac','dogma_paired_bc','dogma_paired_full','dogma_single_full','dogma_single')) {
  print(task)
  o[['task']] = task
  
  config <- parseTOML("configs/data.toml")[[o$task]]
  subset_names <- basename(config$raw_data_dirs)
  subset_ids <- sapply(seq_along(subset_names) - 1, toString)
  input_dirs <-
    pj(
      "result",
      o$task,
      o$exp,
      "default",
      "predict",
      "sp_00001899",
      paste0("subset_", subset_ids)
    )
  output_dir <- pj("result", "analysis", o$task, o$method)
  mkdir(output_dir, remove_old = F)
  
  
  pp_dir <- pj("data", "processed", o$task)
  
  K <- parseTOML("configs/model.toml")[["default"]]$dim_c
  
  
  #Load model outputs
  ######
  rna_bc_list <- list()
  atac_bc_list <- list()
  adt_bc_list <- list()
  cell_name_list <- list()
  S <- length(subset_names)
  for (i in seq_along(subset_names)) {
    subset_name <- subset_names[i]
    z_dir  <- pj(input_dirs[i], "z", "joint")
    rna_bc_dir  <- pj(input_dirs[i], "x", "rna")
    atac_bc_dir <- pj(input_dirs[i], "x", "atac")
    adt_bc_dir  <- pj(input_dirs[i], "x", "adt")
    fnames <- dir(path = z_dir, pattern = ".csv$")
    fnames <- str_sort(fnames, decreasing = F)
    
    rna_bc_subset_list <- list()
    atac_bc_subset_list <- list()
    adt_bc_subset_list <- list()
    N <- length(fnames)
    for (n in seq_along(fnames)) {
      message(paste0("Loading Subset ", i, "/", S, ", File ", n, "/", N))
      if (file.exists(file.path(rna_bc_dir, fnames[n]))) {
        rna_bc_subset_list[[n]] <-
          read.csv(file.path(rna_bc_dir, fnames[n]), header = F)
      }
      if (file.exists(file.path(atac_bc_dir, fnames[n]))) {
        atac_bc_subset_list[[n]] <-
          read.csv(file.path(atac_bc_dir, fnames[n]), header = F)
        
      }
      if (file.exists(file.path(adt_bc_dir, fnames[n]))) {
        adt_bc_subset_list[[n]] <-
          read.csv(file.path(adt_bc_dir, fnames[n]), header = F)
      }
    }
    rna_bc_list[[subset_name]] <- bind_rows(rna_bc_subset_list)
    atac_bc_list[[subset_name]] <- bind_rows(atac_bc_subset_list)
    adt_bc_list[[subset_name]] <- bind_rows(adt_bc_subset_list)
    
    cell_name_list[[subset_name]] <-
      read.csv(pj(pp_dir, paste0("subset_", subset_ids[i]),
                  "cell_names.csv"), header = T)[, 2]
  }
  
  
  all_rna <- t(data.matrix(bind_rows(rna_bc_list)))
  rownames(all_rna) <-
    read.csv(pj(pp_dir, "feat", "feat_names_rna.csv"), header = T)[, 2]
  cnames = NULL
  for (st in subset_names) {
    if (ncol(rna_bc_list[[st]]) > 0) {
      print(st)
      cnames <- c(cnames, cell_name_list[[st]])
    }
  }
  colnames(all_rna) = cnames
  all_rna <- SingleCellExperiment(assays = list(counts = all_rna))
  all_rna <- logNormCounts(all_rna)
  decomp <- modelGeneVar(all_rna)
  rna_hvgs <-
    rownames(decomp)[decomp$mean > 0.01 & decomp$p.value <= 0.05]
  length(rna_hvgs)
  if(endsWith(task,'single')){
    rna_hvgs <-
    rownames(decomp)[decomp$mean > 0.01 & decomp$p.value <= 0.1]
  }
 
  
  
  all_atac <- t(data.matrix(bind_rows(atac_bc_list)))
  rownames(all_atac) <-
    read.csv(pj(pp_dir, "feat", "feat_names_atac.csv"), header = T)[, 2]
  cnames = NULL
  for (st in subset_names) {
    if (ncol(atac_bc_list[[st]]) > 0) {
      print(st)
      cnames <- c(cnames, cell_name_list[[st]])
    }
  }
  colnames(all_atac) = cnames
  all_atac <- SingleCellExperiment(assays = list(counts = all_atac))
  all_atac <- logNormCounts(all_atac)
  decomp <- modelGeneVar(all_atac)
  atac_hvgs <-
    rownames(decomp)[decomp$mean > 0.25 & decomp$p.value <= 0.05]
  length(atac_hvgs)
  
  
  
  
  
  all_adt <- t(data.matrix(bind_rows(adt_bc_list)))
  rownames(all_adt) <-
    read.csv(pj(pp_dir, "feat", "feat_names_adt.csv"), header = T)[, 2]
  cnames = NULL
  for (st in subset_names) {
    if (ncol(adt_bc_list[[st]]) > 0) {
      print(st)
      cnames <- c(cnames, cell_name_list[[st]])
    }
  }
  colnames(all_adt) = cnames
  all_adt <- SingleCellExperiment(assays = list(counts = all_adt))
  all_adt <- logNormCounts(all_adt)
  decomp <- modelGeneVar(all_adt)
  adt_hvgs <-
    rownames(decomp)[decomp$mean > 0.01 & decomp$p.value <= 0.1]
  length(adt_hvgs)
  
  if(endsWith(task,'single')){
    adt_hvgs = rownames(all_adt)
  }
  
  
  merged_mt = list()
  rna_feat = read.csv(pj(pp_dir, "feat", paste0("feat_names_rna.csv")), header = T)[, 2]
  adt_feat = read.csv(pj(pp_dir, "feat", paste0("feat_names_adt.csv")), header = T)[, 2]
  atac_feat = read.csv(pj(pp_dir, "feat", paste0("feat_names_atac.csv")), header = T)[, 2]
  
  mod_list = list()
  for (st in subset_names) {
    print(st)
    bind_list = list()
    mods = NULL
    if (nrow(rna_bc_list[[st]]) > 0) {
      t_ = rna_bc_list[[st]][, rna_feat %in% rna_hvgs]
      colnames(t_) = paste('rna', ":", colnames(t_), sep = "")
      bind_list[['rna']] = t_
      mods = c(mods, 'rna')
    }
    if (nrow(adt_bc_list[[st]]) > 0) {
      t_ = adt_bc_list[[st]][, adt_feat %in% adt_hvgs]
      colnames(t_) = paste('adt', ":", colnames(t_), sep = "")
      bind_list[['adt']] = t_
      mods = c(mods, 'adt')
    }
    if (nrow(atac_bc_list[[st]]) > 0) {
      t_ = atac_bc_list[[st]][, atac_feat %in% atac_hvgs]
      colnames(t_) = paste('atac', ":", colnames(t_), sep = "")
      bind_list[['atac']] = t_
      mods = c(mods, 'atac')
    }
    mod_list[[st]] = mods
    merged_mt[[st]] = t(data.matrix(bind_cols(bind_list)))
    rnames = NULL
    for (mod in c('rna', 'adt', 'atac')) {
      if (mod %in% names(bind_list)) {
        if (mod == 'rna') {
          names_ = rna_feat[rna_feat %in% rna_hvgs]
        }
        if (mod == 'adt') {
          names_ = adt_feat[adt_feat %in% adt_hvgs]
        }
        if (mod == 'atac') {
          names_ = atac_feat[atac_feat %in% atac_hvgs]
        }
        rnames <- c(rnames, names_)
      }
    }
    rownames(merged_mt[[st]]) = rnames
    colnames(merged_mt[[st]]) = paste(st, ":", cell_name_list[[st]], sep = "")
  }
  
  
  
  pbmc <- MultiAssayExperiment(experiments = merged_mt)
  #upsetSamples(pbmc)
  experiments(pbmc)
  #mosaicDataUpSet(merged_mt, plot = FALSE)
  mdt = mosaicDataTopology(merged_mt)
  print(mdt)
  #plot(mdt)
  
  mod_num = 0
  ref = ''
  for (st in subset_names) {
    if (length(mod_list[[st]]) > mod_num) {
      mod_num = length(mod_list[[st]])
      ref = st
    }
  }
  if (mod_num == 1) {
    print('no modality overlap between batches')
    
    adt_syms = read.csv('./feat_names_adt.withSymbol.csv', sep = '\t')
    adt_syms$symbol = as.character(map(strsplit(adt_syms$symbol,','),1))
    rownames(adt_syms) = adt_syms$x
    
    annotation <- GetGRangesFromEnsDb(EnsDb.Hsapiens.v86)
    seqlevelsStyle(annotation) <- "UCSC"
    genome(annotation) <- "hg38"
    atac_obj <-
      CreateChromatinAssay(
        counts = all_atac@assays@data$counts,
        genome = 'hg38',
        annotation = annotation
      )
    for (st in subset_names) {
      print(st)
      if (c('rna') %in% mod_list[[st]]) {
        ref = st
      }
      #adt to gene
      rn = rownames(merged_mt[[st]])
      rn[rn %in% adt_syms$x] = adt_syms[rn[rn %in% adt_syms$x], 'symbol']
      #atac to gene
      regions = rn[grepl('^chr', rn)]
      if (length(regions) > 0) {
        cfs =  ClosestFeature(atac_obj, regions = regions)
        #cfs[cfs$distance < 2000, 'query_region'] = cfs[cfs$distance < 2000, ]$gene_name
        #replaced_genes = cfs$query_region
        replaced_genes = cfs$gene_name
        rn[grepl('^chr', regions)] = replaced_genes
      }
      
      #remove duplicate
      rownames(merged_mt[[st]]) = rn
      merged_mt[[st]] = merged_mt[[st]][!duplicated(rn), ]
      
    }
    mdt = mosaicDataTopology(merged_mt)
    print(mdt)
    plot(mdt)
  }
  stab = stabMap(merged_mt,
                 reference_list = c(ref),
                 plot = FALSE,
                 ncomponentsReference=32,
                 ncomponentsSubset=32)
  dim(stab)
  stab[1:5, 1:5]
  stab_umap = calculateUMAP(t(stab))
  cols = NULL
  for (i in seq_along(rownames(stab_umap))) {
    if (substring(rownames(stab_umap)[i],
                  first = 1,
                  last = 8) == 'lll_stim') {
      cols = c(cols, 'red')
    }
    if (substring(rownames(stab_umap)[i],
                  first = 1,
                  last = 8) == 'lll_ctrl') {
      cols = c(cols, 'pink')
    }
    if (substring(rownames(stab_umap)[i],
                  first = 1,
                  last = 8) == 'dig_stim') {
      cols = c(cols, 'blue')
    }
    if (substring(rownames(stab_umap)[i],
                  first = 1,
                  last = 8) == 'dig_ctrl') {
      cols = c(cols, 'green')
    }
  }
  spt = strsplit(rownames(stab_umap), ':')
  batches = unlist(spt)[c(TRUE, FALSE)]
  cells = unlist(spt)[c(FALSE, TRUE)]
  #plot(stab_umap,
  #     pch = 16,
  #     cex = 0.3,
  #     col = factor(cols))
  
  rownames(stab) = cells
  #aggregate(stab, list(row.names(stab)), mean)
  #mean_stab = do.call(rbind, by(stab, row.names(stab), FUN = colMeans))
  write.csv(stab, file = pj(output_dir, paste0('stabmap.ref.', ref, '.csv')))
  
  
}
