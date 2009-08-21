gpuMi <- function(x, y = NULL, bins = 2, splineOrder = 1)
{
	x <- as.matrix(x)
	if(!is.null(y)) {
		y <- as.matrix(y)
	}

	bins <- as.integer(bins)
	splineOrder <- as.integer(splineOrder)

	cols <- nrow(x)

	a <- as.single(x)
	rowsA <- ncol(x)

	if(is.null(y)) {
		b <- a
		rowsB <- ncol(x)
	} else {
		b <- as.single(y)
		rowsB <- ncol(y)
	}

	mutualInfo <- single(rowsB * rowsA)

	if(is.null(y)) {
		cCall <- .C("rBSplineMutualInfoSingle",
			cols, bins, splineOrder, rowsA, a,
			mi = mutualInfo)
		mutualInfo <- cCall$mi
	} else {
		cCall <- .C("rBSplineMutualInfo", cols,
			bins, splineOrder, rowsA, a, rowsB, b,
			mi = mutualInfo)
		mutualInfo <- cCall$mi
	}

	mutualInfo <- matrix(mutualInfo, rowsB, rowsA)
	if(is.null(y)) {
		rownames(mutualInfo) <- colnames(x)
	} else {
		rownames(mutualInfo) <- colnames(y)
	}
	colnames(mutualInfo) <- colnames(x)
	return(mutualInfo)
}
