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