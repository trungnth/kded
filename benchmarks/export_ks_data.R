# Writes the three open data sets of the R package ks used by the benchmark
# (quake, hsct, air) to data/*.csv. Run once from this folder:
#   Rscript export_ks_data.R
suppressMessages(library(ks))
data(quake); data(hsct); data(air)
dir.create("data", showWarnings = FALSE)
q <- quake[!is.na(quake$depth), c("long", "lat", "depth")]
write.csv(q, "data/quake3.csv", row.names = FALSE)
write.csv(hsct, "data/hsct.csv", row.names = FALSE)
write.csv(air[, c("date", "time", "no2", "pm10", "co2", "temp", "humi")],
          "data/air.csv", row.names = FALSE)
cat("ks", as.character(packageVersion("ks")), ": wrote data/quake3.csv, data/hsct.csv, data/air.csv\n")
