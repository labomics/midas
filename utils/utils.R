library(argparse)


pj <- file.path


prt <- function(...) {
    cat(paste0(..., "\n"))
}



mkdir <- function(directory, remove_old = F) {
    if (remove_old) {
        if (dir.exists(directory)) {
             prt("Removing directory ", directory)
             unlink(directory, recursive = T)
        }
    }
    if (!dir.exists(directory)) {
        dir.create(directory, recursive = T)
    }
}


mkdirs <- function(directories, remove_old = F) {
    for (directory in directories) {
        mkdir(directory, remove_old = remove_old)
    }
}


random_round <- function(mat) {
    mat_floor <- floor(mat)
    res <- mat - mat_floor
    res[] <- rbinom(n = nrow(res) * ncol(res), size = 1, prob = res)
    mat <- mat_floor + res
    mode(mat) <- "integer"
    return(mat)
}