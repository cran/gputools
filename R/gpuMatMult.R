gpuMatMult <- function(a, b) 
{
	a <- as.matrix(a)
	b <- as.matrix(b)

	results <- .C("RgpuMatMult", PACKAGE = "gputools",
		as.single(a), as.integer(nrow(a)), as.integer(ncol(a)),
		as.single(b), as.integer(nrow(b)), as.integer(ncol(b)),
		output = single(nrow(a)*ncol(b)))

	matrix(results$output, nrow(a), ncol(b))
}
