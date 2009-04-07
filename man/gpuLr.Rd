\name{gpuLr}
\alias{gpuLr}
\title{A Toy Logistic Regression Routine}

\description{
	This function is an demonstration of logistic regression.  The matrix QR
	decomposition implemented by this function is not quite up to snuff.  The
	regression is not numerically stable and doesn't handle inputs that give 
	rise to rank deficient matrices.  We hope to release a better version of
	this in the future.
}

\usage{
	gpuLr(obs, outcomes, epsilon=0.00001, ridge=0.00001, maxiter=200)
}

\arguments{
	\item{obs}{a matrix of floating point numbers.  Each column is an observed
		set of parameters (inputs) for the logistic function.}
	\item{outcomes}{a vector of floating point numbers.  Each value is an
		observed output when the corresponding column of the 'obs' argument
		is input into the logistic function.}
	\item{epsilon}{a floating point number.  When the sum of squares of the 
		difference between outputs
		of the logistic functions on input of obs is less than the number of 
		columns of obs times epsilon squared, the regression is declared to have
		converged and the iteration stops.}
	\item{ridge}{a floating point number.  Amount of error to introduce during 
		the iterations of the regression to aid convergence.}
	\item{maxiter}{an integer.  Maximum number of iterations that the regression
		will be allowed to execute without converging.}
}

\value{
	a vector of floating point numbers of length 'nrow(obs)'.  The coefficients 
	of the logistic function fitted to 'obs' and 'outcomes'.
}

\seealso{
	glm
}

\examples{
numParameters <- 5
numExperiments <- 5
observationMat <- matrix(runif(numParameters*numExperiments), numParameters, 
	numExperiments)
obsOutcomes <- runif(numExperiments)
for(i in 1:numExperiments) \{
	if(obsOutcomes[i] > 0.5) \{
		obsOutcomes[i] <- 1
	\} else \{
		obsOutcomes[i] <- 0
	\}
\}
gpuLr(observationMat, obsOutcomes)
}

\keyword{models}
\keyword{regression}
\keyword{nonlinear}
