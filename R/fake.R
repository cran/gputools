gpuSvmTrain <- function(y, x, C = 10, kernelWidth = 0.125, eps = 0.5, 
	stoppingCrit = 0.001, isRegression = FALSE)
{
	if(TRUE) {
		stop("svm functions not implemented in device emulation mode")
	}

	# input
	m <- as.integer(nrow(x))
	n <- as.integer(ncol(x))

	y <- as.single(y)
	x <- as.single(x)

	C <- as.single(C)
	kernelWidth <- as.single(kernelWidth)
	eps <- as.single(eps)
	stoppingCrit <- as.single(stoppingCrit)

	regressionBit <- as.integer(0)
	if(isRegression) {
		regressionBit <- as.integer(1)
	}

	# output
	alpha <- single(m)
	beta <- single(1)
	numSvs <- integer(1)
	numPosSvs <- integer(1)

	if(isRegression) {
		call1 <- .C("R_SVRTrain", alpha = alpha, beta = beta,
			y, x,
			C, kernelWidth, eps, 
			m, n, 
			stoppingCrit, numSvs = numSvs)

		numSvs <- call1$numSvs
		numPosSvs <- integer(1)
		alpha <- call1$alpha
		beta <- call1$beta
					
		call2 <- .C("R_produceSupportVectors", regressionBit, m, n, 
			call1$numSvs, numPosSvs, x, y, call1$alpha,
			svCoefficients = single(numSvs),
			supportVectors = single(numSvs * n))
	} else {
		call1 <- .C("R_SVMTrain", alpha = alpha, beta = beta,
			y, x,
			C, kernelWidth,
			m, n, 
			stoppingCrit, numSvs = numSvs, numPosSvs = numPosSvs)
			
		numSvs <- call1$numSvs
		numPosSvs <- call1$numPosSvs
		alpha <- call1$alpha
		beta <- call1$beta

		call2 <- .C("R_produceSupportVectors", regressionBit, m, n,
			numSvs, numPosSvs, x, y, alpha,
			svCoefficients = single(numSvs),
			supportVectors = single(numSvs * n))
	}
	list(supportVectors = matrix(call2$supportVectors, numSvs, n),
		svCoefficients = call2$svCoefficients, svOffset = beta)
}

gpuSvmPredict <- function(data, supportVectors, svCoefficients, svOffset,
	kernelWidth = 0.125, isRegression = FALSE)
{
	if(TRUE) {
		stop("svm functions not implemented in device emulation mode")
	}

	m <- as.integer(nrow(data))
	k <- as.integer(ncol(data))
	n <- as.integer(length(svCoefficients))

	data <- as.single(data)
	supportVectors <- as.single(supportVectors)
	svCoefficients <- as.single(svCoefficients)
	svOffset <- as.single(svOffset)
	kernelWidth <- as.single(kernelWidth)

	predictions <- single(m)

	if(isRegression) {
		isRegression <- as.integer(1)
	} else {
		isRegression <- as.integer(0)
	}	

	classes <- .C("R_GPUPredictWrapper", m, n, k, 
		kernelWidth, data, supportVectors, svCoefficients,
		output = predictions,
		svOffset, isRegression)

	classes$output
}

getAucEstimate <- function(classes, scores)
{
	if(TRUE) {
		stop("svm functions not implemented in device emulation mode")
	}

	classes <- as.double(classes)
	n <- as.integer(length(classes))
	c <- .C("RgetAucEstimate", NAOK = TRUE, n, classes, scores, 
		auc = double(1))
	c$auc
}
