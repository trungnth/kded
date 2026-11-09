# Bandwidth selectors of the R package ks (and bw.SJ of base R for d = 1) on
# the training halves written by
#   python run_benchmark.py --export-splits splits
# Usage (from this folder):
#   Rscript ks_selectors.R splits results/local/ks_H.csv [first last]
# The optional first/last select a range of lines of splits/keys.txt, so the
# work can be spread over a job array. Output, one line per selector:
#   key, method, seconds, status, H (column-major)
# Then score in Python:
#   python run_benchmark.py --score-ks results/local/ks_H.csv --out results/local/ks.jsonl
suppressMessages(library(ks))
a <- commandArgs(trailingOnly = TRUE)
folder <- if (length(a) >= 1) a[1] else "splits"
outf <- if (length(a) >= 2) a[2] else "results/local/ks_H.csv"
dir.create(dirname(outf), recursive = TRUE, showWarnings = FALSE)
keys <- readLines(file.path(folder, "keys.txt"))
if (length(a) >= 4) keys <- keys[as.integer(a[3]):min(as.integer(a[4]), length(keys))]
done <- if (file.exists(outf)) unique(paste(sub(",.*", "", readLines(outf)))) else character(0)

sel1 <- list(ks_hpi = function(x) hpi(x)^2, ks_hscv = function(x) hscv(x)^2,
             ks_hlscv = function(x) hlscv(x)^2, r_bwsj = function(x) bw.SJ(x)^2)
seld <- list(ks_hpi = Hpi, ks_hscv = Hscv, ks_hlscv = Hlscv)

for (f in keys) {
  key <- sub("\\.csv$", "", f)
  if (key %in% done) next
  x <- as.matrix(read.csv(file.path(folder, f), header = FALSE))
  d <- ncol(x)
  sel <- if (d == 1) sel1 else seld
  if (d == 1) x <- x[, 1]
  lines <- character(0)
  for (m in names(sel)) {
    t0 <- proc.time()[3]
    H <- tryCatch(as.matrix(sel[[m]](x)), error = function(e) conditionMessage(e))
    dt <- proc.time()[3] - t0
    if (is.character(H)) {
      st <- gsub("[,\n]", " ", paste("error:", substr(H, 1, 120)))
      lines <- c(lines, paste(key, m, format(dt), st, sep = ","))
    } else {
      lines <- c(lines, paste(c(key, m, format(dt), "ok",
                                format(as.vector(H), digits = 17)), collapse = ","))
    }
  }
  cat(lines, file = outf, sep = "\n", append = TRUE)
  cat(key, if (d == 1) length(x) else nrow(x), "done\n")
}
