gpuQr <- function(x, tol = 1e-07) {

	x <- as.matrix(x)
	if(is.complex(x)) {
		stop("complex gpuQR not yet supported")
	}

	n <- nrow(x)
	p <- ncol(x)

	mode(x) <- 'single'

	res <- .C("rGetQRDecompPacked",
		as.integer(n),
		as.integer(p),
		as.single(tol),
		qr = x,
		pivot = as.integer(1L:p),
		qraux = single(p),
		rank = integer(1L)
	)

	if(!is.null(cn <- colnames(x)))
		colnames(res$qr) <- cn[res$pivot]

	class(res) <- "qr"
	res
}

# solve for b: xb = y
# gpuSolve <- function(x, y)
# {
#	x <- as.matrix(x)
#	y <- as.matrix(y)
#
#	m <- nrow(x)
#	n <- ncol(x)
#	fcall <- .C("RqrSolver", as.integer(m), as.integer(n), as.single(x), 
#		as.single(y), solution = single(n))
#	return(fcall$solution)
#}
