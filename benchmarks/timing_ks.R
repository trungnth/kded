# Computing time of the R package ks: plug-in selector (hpi / Hpi) plus the
# estimate on the grid of timing.py (1024, 128^2, 64^3 points), same normal
# mixture as timing.py (different random numbers). One thread.
# Usage: Rscript timing_ks.R [results/timing/timing_ks.jsonl] [timeout_seconds]
suppressMessages(library(ks))
a <- commandArgs(trailingOnly = TRUE)
outf <- if (length(a) >= 1) a[1] else "results/timing/timing_ks.jsonl"
dir.create(dirname(outf), recursive = TRUE, showWarnings = FALSE)
tmax <- if (length(a) >= 2) as.numeric(a[2]) else 600
grid <- c(1024, 128, 64)
for (d in 1:3) {
  for (N in c(1e3, 1e4, 1e5, 1e6)) {
    times <- c()
    for (rep in 1:3) {
      set.seed(rep)
      X <- matrix(rnorm(N * d), N, d)
      k <- runif(N) < 0.5
      X[k, ] <- X[k, ] * 0.5 + 2
      if (d == 1) X <- X[, 1]
      t0 <- proc.time()[3]
      if (d == 1) { h <- hpi(X); f <- kde(X, h = h, gridsize = grid[d]) }
      else { H <- Hpi(X); f <- kde(X, H = H, gridsize = rep(grid[d], d)) }
      times <- c(times, proc.time()[3] - t0)
      if (times[length(times)] > tmax) break
    }
    line <- sprintf('{"method": "ks_hpi", "d": %d, "N": %d, "status": "ok", "time": %.6g, "times": [%s]}',
                    d, as.integer(N), median(times), paste(format(times), collapse = ", "))
    cat(line, "\n", file = outf, append = TRUE)
    cat(line, "\n")
    if (median(times) > tmax) break
  }
}
