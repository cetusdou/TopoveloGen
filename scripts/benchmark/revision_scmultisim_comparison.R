#!/usr/bin/env Rscript

root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1])), "..", ".."))
.libPaths(c(file.path(root, ".r-methods"), .libPaths()))
suppressPackageStartupMessages(library(scMultiSim))
data(GRN_params_100)

out <- file.path(root, "data", "results", "revision_scmultisim_comparison")
dir.create(out, recursive=TRUE, showWarnings=FALSE)

for (seed in c(11, 22, 33)) {
  set.seed(seed)
  options <- list(GRN=GRN_params_100, num.cells=360, num.genes=160, num.cifs=20,
                  tree=Phyla3(), do.velocity=TRUE)
  result <- sim_true_counts(options)
  cat("seed", seed, "fields", paste(names(result), collapse=","), "\n")
  for (field in names(result)) {
    value <- result[[field]]
    if (is.matrix(value)) cat(field, paste(dim(value), collapse="x"), "\n")
  }
  cat("kinetic fields", paste(names(result$kinetic_params), collapse=","), "\n")
  write.csv(t(result$counts), file.path(out, sprintf("spliced_seed%d.csv", seed)), row.names=FALSE)
  write.csv(t(result$unspliced_counts), file.path(out, sprintf("unspliced_seed%d.csv", seed)), row.names=FALSE)
  write.csv(t(result$velocity), file.path(out, sprintf("velocity_seed%d.csv", seed)), row.names=FALSE)
  write.csv(result$cell_meta, file.path(out, sprintf("cell_meta_seed%d.csv", seed)), row.names=FALSE)
  saveRDS(result, file.path(out, sprintf("scmultisim_seed%d.rds", seed)))
}
