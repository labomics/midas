source("/root/workspace/code/sc-transformer/utils/utils.R")
library(Seurat)
library(SeuratDisk)
library(Signac)
library(future)
library(EnsDb.Hsapiens.v86)
library(BSgenome.Hsapiens.UCSC.hg38)
library(dplyr)
library(ggplot2)
library(Matrix)
library(purrr)
library(stringr)
library(GenomicRanges)
library(RcppTOML)
library(ymlthis)
library(argparse)
library(RColorBrewer)
set.seed(1234)
plan("multicore", workers = 64)
options(future.globals.maxSize = 100 * 1024^3) # for 100 Gb RAM
options(future.seed = T)


gen_atac <- function(frag_path, min_cells = 5) {
    # call peaks using MACS2
    system(paste0("tabix -f -p bed ", frag_path))
    frags <- CreateFragmentObject(frag_path)
    peaks <- CallPeaks(frags)
    peaks@seqnames
    # remove peaks on non-autosomes and in genomic blacklist regions
    peaks <- keepStandardChromosomes(peaks, pruning.mode = "coarse")
    peaks <- peaks[!(peaks@seqnames %in% c("chrX", "chrY"))]
    peaks <- subsetByOverlaps(x = peaks, ranges = blacklist_hg38_unified, invert = TRUE)
    # quantify counts in each peak
    atac_counts <- FeatureMatrix(
        fragments = frags,
        features = peaks
    )
    # # add in the atac-seq data, only use peaks in standard chromosomes
    # grange <- StringToGRanges(rownames(atac_counts))
    # grange_use <- seqnames(grange) %in% standardChromosomes(grange)
    # atac_counts <- atac_counts[as.vector(grange_use), ]
    # get gene annotations for hg38
    annotation <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
    seqlevelsStyle(annotation) <- "UCSC"
    genome(annotation) <- "hg38"
    # create atac assay and add it to the object
    atac_assay <- CreateChromatinAssay(
        counts = atac_counts,
        min.cells = min_cells,
        genome = 'hg38',
        fragments = frags,
        annotation = annotation
    )
    atac <- CreateSeuratObject(
        counts = atac_assay,
        assay = 'atac',
    )
    atac <- NucleosomeSignal(atac)
    atac <- TSSEnrichment(atac)
    return(atac)
}


gen_rna <- function(rna_counts, min_cells = 3) {
    rna <- CreateSeuratObject(
        counts = rna_counts,
        min.cells = min_cells,
        assay = "rna"
    )
    rna[["percent.mt"]] <- PercentageFeatureSet(rna, pattern = "^MT-")
    return(rna)
}


remove_sparse_genes <- function(obj, assay = "rna", min_cell_percent = 1, kept_genes = NULL) {
    assay_ <- DefaultAssay(obj)
    DefaultAssay(obj) <- assay
    min_cells <- 0.01 * min_cell_percent * ncol(obj)
    mask <- rowSums(obj[[assay]]@counts > 0) > min_cells & rowSums(obj[[assay]]@counts) > 2 * min_cells
    feats <- rownames(obj[[assay]]@counts)[mask]
    if (!is.null(kept_genes)) {
        feats <- union(feats, kept_genes)
    }
    obj <- subset(obj, features = feats)
    DefaultAssay(obj) <- assay_
    return(obj)
}


gen_adt <- function(adt_counts) {
    # rename features
    feat <- unlist(map(rownames(adt_counts), tolower))
    feat <- unlist(map(feat, gsub, pattern = "-|_|\\(|\\)|/", replacement = "."))
    feat <- unlist(map(feat, gsub, pattern = "^cd3$", replacement = "cd3.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd4$", replacement = "cd4.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd11b$", replacement = "cd11b.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd26$", replacement = "cd26.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd38$", replacement = "cd38.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd56$", replacement = "cd56.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd56.ncam.$", replacement = "cd56.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd56.ncam.recombinant$", replacement = "cd56.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd57.recombinant$", replacement = "cd57"))
    feat <- unlist(map(feat, gsub, pattern = "^cd90.thy1.$", replacement = "cd90"))
    feat <- unlist(map(feat, gsub, pattern = "^cd112.nectin.2.$", replacement = "cd112"))
    feat <- unlist(map(feat, gsub, pattern = "^cd117.c.kit.$", replacement = "cd117"))
    feat <- unlist(map(feat, gsub, pattern = "^cd138.1.syndecan.1.$", replacement = "cd138.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd155.pvr.$", replacement = "cd155"))
    feat <- unlist(map(feat, gsub, pattern = "^cd269.bcma.$", replacement = "cd269"))
    feat <- unlist(map(feat, gsub, pattern = "^clec2$", replacement = "clec1b"))
    feat <- unlist(map(feat, gsub, pattern = "^cadherin11$", replacement = "cadherin"))
    feat <- unlist(map(feat, gsub, pattern = "^folate.receptor$", replacement = "folate"))
    feat <- unlist(map(feat, gsub, pattern = "^notch.1$", replacement = "notch1"))
    feat <- unlist(map(feat, gsub, pattern = "^notch.2$", replacement = "notch3"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.a.b$", replacement = "tcrab"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.2$", replacement = "tcrab"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.g.d$", replacement = "tcrgd"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.1$", replacement = "tcrgd"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.va7.2$", replacement = "tcr.v.7.2"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.va24.ja18$", replacement = "tcr.v.24.j.18"))
    feat <- unlist(map(feat, gsub, pattern = "^vegfr.3$", replacement = "vegfr3"))
    # feat <- unlist(map(feat, gsub, pattern = "^igg1.k.isotype.control$", replacement = "rat.igg1k.isotypectrl"))
    rownames(adt_counts) <- feat
    # remove features
    adt_counts <- adt_counts[-grep("igg", rownames(adt_counts)), ]
    # create adt object
    adt <- CreateSeuratObject(
      counts = adt_counts,
      assay = "adt"
    )
    return(adt)
}


preprocess <- function(output_dir, atac = NULL, rna = NULL, adt = NULL) {
    # preprocess and save data
    if (!is.null(atac)) {
        atac <- RunTFIDF(atac) %>%
                FindTopFeatures(min.cutoff = "q0")
        SaveH5Seurat(atac, pj(output_dir, "atac.h5seurat"), overwrite = TRUE)
    }

    if (!is.null(rna)) {
        rna <- NormalizeData(rna) %>%
               FindVariableFeatures(nfeatures = 4000) %>%
               ScaleData()
        SaveH5Seurat(rna, pj(output_dir, "rna.h5seurat"), overwrite = TRUE)
    }

    if (!is.null(adt)) {
        VariableFeatures(adt) <- rownames(adt)
        adt <- NormalizeData(adt, normalization.method = "CLR", margin = 2) %>%
               ScaleData()
        SaveH5Seurat(adt, pj(output_dir, "adt.h5seurat"), overwrite = TRUE)
    }
}


get_adt_genes <- function(file_path = "configs/adt_rna_correspondence.csv") {
    adt_genes_raw <- read.csv(file_path, sep = "\t")[["symbol"]]
    adt_genes <- vector()
    for (gene in adt_genes_raw) {
        if (gene %in% c("not_found", "")) {
            next
        } else if (grepl(",", gene)) {
            adt_genes <- c(adt_genes, strsplit(gene, split = ",")[[1]])
        } else {
            adt_genes <- c(adt_genes, gene)
        }
    }
    return(unique(adt_genes))
}


plt_size <- function(w, h) {
     options(repr.plot.width = w, repr.plot.height = h)
}


dim_plot <- function(obj, w, h, reduction = NULL, split.by = NULL, group.by = NULL,
    label = F, repel = F, label.size = 4, pt.size = NULL, order = NULL, shuffle = T, cols = NULL,
    save_path = NULL, legend = T, title = NULL, display = T, no_axes = F, return_plt = F,
    border = F, raster = F, rater_dpi = 250) {

    plt_size(w = w, h = h)
    plt <- DimPlot(obj, reduction = reduction, split.by = split.by, group.by = group.by,
    label = label, repel = repel, label.size = label.size, pt.size = pt.size, shuffle = shuffle,
    order = order, cols = cols, raster = raster, raster.dpi = c(rater_dpi, rater_dpi))
    if (!legend) {
        plt <- plt + NoLegend()
    }

    if (!is.null(title)) {
        plt <- plt + ggtitle(title) + theme(plot.title = element_text(face = "plain", size = 40))
        title_margin <- 8
    } else {
        plt <- plt + theme(plot.title = element_blank())
        title_margin <- 0
    }

    if (no_axes) {
        plt <- plt + NoAxes()
    }

    if (border) {
        plt <- plt + theme(panel.border = element_rect(color = "black", linewidth = 1),
                           axis.ticks.length = unit(0, "pt"), plot.margin = margin(title_margin, 0, 0, 0))
    }

    if (!is.null(save_path)) {
        ggsave(plot = plt, file = paste0(save_path, ".png"), width = w, height = h, limitsize = F)
        ggsave(plot = plt, file = paste0(save_path, ".pdf"), width = w, height = h, limitsize = F)
    }

    if (display) {
        plt
    }

    if (return_plt) {
        return(plt)
    }
}


# https://mokole.com/palette.html
# col_9 <- brewer.pal(n = 9, name = "Set1")



col_27 <- c("#8b4513", "#6b8e23", "#483d8b", "#bc8f8f", "#008080", "#000080", "#daa520", "#8fbc8f", "#8b008b",
            "#b03060", "#ff0000", "#00ff00", "#00fa9a", "#8a2be2", "#dc143c", "#00ffff", "#00bfff", "#0000ff",
            "#adff2f", "#ff7f50", "#ff00ff", "#1e90ff", "#f0e68c", "#ffff54", "#add8e6", "#ff1493", "#ee82ee")

# col_13_ <- c("#00ff00", "#ff0000", "#0000ff", "#ffff00", "#ff69b4", "#00ffff", "#ff00ff", "#008000", "#6495ed",  "#4b0082", "#eee8aa", "#2f4f4f", "#8b4513")
            col_13 <- c("#8FC36D", "#f54646", "#4472c4", "#fff300", "#ff69b4", "#ff00ff", "#14e6e6", "#008000", "#82B4ed",  "#D4aaff", "#eee8aa", "#2f4f4f", "#ad6800")


# col_9  <- c("#ff4500", "#006400", "#0000ff", "#ffd700", "#ff1493", "#00ffff", "#4169e1", "#00ff00", "#bc8f8f")

# col_8 <- c("#00ff00", "#ff0000", "#0000ff", "#006400", "#c71585", "#00ffff", "#1e90ff", "#ffd700")
col_8 <- c("#8FC36D", "#f54646", "#4472c4", "#ff00ff", "#82B4ed", "#D4aaff", "#008000", "#fff300")

col_4  <- c("#00ff00", "#ff0000", "#0000ff", "#87cefa")
# qual_col_pals <- brewer.pal.info[brewer.pal.info$category == 'qual',]
# col_max <- unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))


dim_reduc <- function(obj, atac = "atac", rna = "rna", adt = "adt") {
    DefaultAssay(obj) <- atac
    obj <-  RunTFIDF(obj) %>%
            FindTopFeatures(min.cutoff = "q25") %>%
            RunSVD(reduction.name = "lsi")

    DefaultAssay(obj) <- rna
    VariableFeatures(obj) <- rownames(obj)
    obj <-  NormalizeData(obj) %>%
            # FindVariableFeatures(nfeatures = 2000) %>%
            ScaleData() %>%
            RunPCA(reduction.name = "pca_rna", verbose = F)

    DefaultAssay(obj) <- adt
    VariableFeatures(obj) <- rownames(obj)
    obj <-  NormalizeData(obj, normalization.method = "CLR", margin = 2) %>%
            ScaleData() %>%
            RunPCA(reduction.name = "pca_adt", verbose = F)

    return(obj)
}


rename_task <- function(task) {
    for (data in c("dogma", "teadog")) {
        task <- gsub(paste0(data, "_full"       ), paste0(data, "-full"),
                gsub(paste0(data, "_single"     ), paste0(data, "-diagonal"),
                gsub(paste0(data, "_paired_a"   ), paste0(data, "-paired-a"),
                gsub(paste0(data, "_paired_b"   ), paste0(data, "-paired-b"),
                gsub(paste0(data, "_paired_c"   ), paste0(data, "-paired-c"),
                gsub(paste0(data, "_paired_ab"  ), paste0(data, "-paired-ab"),
                gsub(paste0(data, "_paired_ac"  ), paste0(data, "-paired-ac"),
                gsub(paste0(data, "_paired_bc"  ), paste0(data, "-paired-bc"),
                gsub(paste0(data, "_paired_abc" ), paste0(data, "-paired-abc"),
                gsub(paste0(data, "_paired_full"), paste0(data, "-paired+full"),
                gsub(paste0(data, "_single_full"), paste0(data, "-diagonal+full"),
                gsub(paste0(data, "_single_atac"), paste0(data, "-atac"),
                gsub(paste0(data, "_single_rna" ), paste0(data, "-rna"),
                gsub(paste0(data, "_single_adt" ), paste0(data, "-adt"), task))))))))))))))
    }
    task <- gsub("_transfer", " (model transfer)", task)
    return(task)
}