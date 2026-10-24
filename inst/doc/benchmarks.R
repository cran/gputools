## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")
bench <- read.csv("benchmarks.csv")
machine <- read.csv("benchmarks-machine.csv")

## ----machine, echo = FALSE----------------------------------------------------
knitr::kable(machine, col.names = c("", ""))

## ----results, echo = FALSE----------------------------------------------------
tab <- bench[!is.na(bench$cpu), ]
speedup <- function(gpu) ifelse(is.na(gpu), "",
                                sprintf("%.1fx", tab$cpu / gpu))
shown <- data.frame(
  Task = tab$task, Size = tab$size,
  `base R` = sprintf("%.3f", tab$cpu),
  `GPU single` = ifelse(is.na(tab$single), "",
                        sprintf("%.3f", tab$single)),
  `speed-up` = speedup(tab$single),
  `GPU double` = sprintf("%.3f", tab$double),
  `speed-up ` = speedup(tab$double),
  check.names = FALSE)
knitr::kable(shown, align = "llrrrrr")

## ----chart, echo = FALSE, fig.width = 7, fig.height = 4.5, fig.alt = "Speed-up of each gputools function over base R in single and double precision, on a log scale; values above 1 mean the GPU is faster."----
s <- tab$cpu / tab$single
d <- tab$cpu / tab$double
labels <- sub(" vs .*", "", tab$task)
op <- par(mar = c(4, 13, 1, 1))
barplot(t(cbind(s, d)), beside = TRUE, horiz = TRUE, log = "x",
        names.arg = labels, las = 1, xlab = "speed-up over base R (log scale)",
        legend.text = c("single", "double"),
        args.legend = list(x = "bottomright", bty = "n"))
abline(v = 1, lty = 2)
par(op)

## ----crossover-data, include = FALSE------------------------------------------
xo <- read.csv("crossover.csv")
xo$speedup <- xo$cpu / xo$single
xo$speedupDouble <- xo$cpu / xo$double
fns <- unique(xo$fn[!is.na(xo$cpu)])
firstAt <- function(f, k) {
  d <- xo[xo$fn == f & !is.na(xo$cpu), ]
  hit <- d$size[d$speedup >= k]
  if (length(hit)) format(min(hit), big.mark = ",", scientific = FALSE) else "not reached"
}
ov <- read.csv("call-overhead.csv")

## ----crossover-table, echo = FALSE--------------------------------------------
what <- vapply(fns, function(f) xo$what[xo$fn == f][1], "")
tested <- vapply(fns, function(f) {
  z <- range(xo$size[xo$fn == f])
  paste(format(z, big.mark = ",", scientific = FALSE), collapse = " to ")
}, "")
knitr::kable(data.frame(
  Function = fns, `Size measured as` = what, Tested = tested,
  `Faster than R from` = vapply(fns, firstAt, "", k = 1),
  `5 times faster from` = vapply(fns, firstAt, "", k = 5),
  check.names = FALSE, row.names = NULL))

## ----crossover-chart, echo = FALSE, fig.width = 7.5, fig.height = 7.5, fig.alt = "Speed-up over base R against data size for each gputools function, single and double precision, on log scales; the dashed line at 1 marks where the GPU starts to win."----
op <- par(mfrow = c(ceiling(length(fns) / 3), 3), mar = c(3.2, 3.2, 2, 0.5),
          mgp = c(2, 0.6, 0), cex = 0.7)
for (f in fns) {
  d <- xo[xo$fn == f & !is.na(xo$cpu), ]
  yr <- range(c(d$speedup, d$speedupDouble, 1), na.rm = TRUE)
  plot(d$size, d$speedup, log = "xy", type = "b", pch = 19, ylim = yr,
       xlab = d$what[1], ylab = "times faster than R", main = f)
  lines(d$size, d$speedupDouble, type = "b", pch = 1, lty = 3)
  abline(h = 1, lty = 2, col = "grey50")
}
par(op)

## ----pairs-data, include = FALSE----------------------------------------------
pp <- read.csv("pairs.csv")
fmt <- function(v) ifelse(is.na(v), "--", sprintf("%.2f", v))

## ----pairs-table, echo = FALSE------------------------------------------------
knitr::kable(data.frame(
  Variables = format(pp$variables, big.mark = ","),
  `Pairs found` = format(pp$pairs, big.mark = ","),
  `cor + which` = fmt(pp$cor), `gpuCor + which` = fmt(pp$gpuCor),
  gpuCorPairs = fmt(pp$gpuCorPairs), `top 10 each` = fmt(pp$top10),
  check.names = FALSE), align = "rrrrrr")

## ----assoc-data, include = FALSE----------------------------------------------
sc <- read.csv("assoc.csv")
secs <- function(v) ifelse(is.na(v), "--", sprintf("%.2f", v))
big <- sc[!is.na(sc$MatrixEQTL), ]
big <- big[which.max(big$snps * big$genes), ]

## ----assoc-table, echo = FALSE------------------------------------------------
knitr::kable(data.frame(
  SNPs = format(sc$snps, big.mark = ","),
  Genes = format(sc$genes, big.mark = ","),
  Tests = format(sc$snps * as.double(sc$genes), big.mark = ",",
                 scientific = FALSE),
  Found = format(sc$found, big.mark = ","),
  `Matrix eQTL` = secs(sc$MatrixEQTL),
  `gpuAssoc single` = secs(sc$single), `gpuAssoc double` = secs(sc$double),
  check.names = FALSE), align = "rrrrrrr")

## ----kmeans-data, include = FALSE---------------------------------------------
km <- read.csv("kmeans.csv")

## ----kmeans-table, echo = FALSE-----------------------------------------------
f <- function(v, d = 2) ifelse(is.na(v), "--", formatC(v, format = "f", digits = d))
knitr::kable(data.frame(
  Points = format(km$points, big.mark = ",", scientific = FALSE),
  Clusters = format(km$clusters, big.mark = ","),
  `kmeans Lloyd` = f(km$lloyd), `kmeans Hartigan-Wong` = f(km$hartiganWong),
  `gpuKmeans single` = f(km$single, 3), `gpuKmeans double` = f(km$double, 3),
  `sum of squares vs H-W` = f(km$wssGpu / km$wssHW, 3),
  check.names = FALSE), align = "rrrrrrr")

## ----perm-data, include = FALSE-----------------------------------------------
pm <- read.csv("perm.csv")
rt <- read.csv("rowtest.csv")
s3 <- function(v) ifelse(is.na(v), "--", formatC(v, format = "f", digits = 3))
s2 <- function(v) ifelse(is.na(v), "--", formatC(v, format = "f", digits = 2))

## ----perm-table, echo = FALSE-------------------------------------------------
knitr::kable(data.frame(
  `x and y variables` = format(pm$variables, big.mark = ","),
  Pairs = format(pm$variables^2, big.mark = ",", scientific = FALSE),
  `cor loop` = s2(pm$corLoop), `BLAS loop` = s2(pm$blasLoop),
  `gpuCorPerm single` = s3(pm$single), `gpuCorPerm double` = s3(pm$double),
  check.names = FALSE), align = "rrrrrr")

## ----rowtest-table, echo = FALSE----------------------------------------------
knitr::kable(data.frame(
  Genes = format(rt$genes, big.mark = ","), Samples = rt$samples,
  `R loop` = s2(rt$cpu), `gpuRowTtest single` = s3(rt$single),
  `gpuRowTtest double` = s3(rt$double),
  `no permutations` = s3(rt$noPermutations),
  check.names = FALSE), align = "rrrrrr")

