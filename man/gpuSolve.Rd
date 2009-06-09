\name{gpuSolve}
\alias{gpuSolve}
\title{Estimate the solution to a matrix vector equation}

\description{
	This function estimates the solution to an equation of the form x * b = y
	where x is a matrix, b is an unkown vector, and y is a known vector.  It
	does much calculation on a GPU.
}

\usage{
	gpuSolve(x, y)
}

\arguments{
	\item{x}{a matrix of floating point numbers.}  
	\item{y}{a vector of floating point numbers of length nrow(x).}
}

\value{
a vector of floating point numbers of length ncol(x) which is an estimate of
the b from x * b = y.
}

\references{
Bjorck, Ake (1996) \emph{Numerical methods for least squares problems}. SIAM.
}

\examples{
	# get some random data of any shape at all
	x <- matrix(runif(1000), 50, 20)
	y <- runif(50)
	b <- gpuSolve(x, y)
	cat("Solution:\n")
	print(b)
	cat("x * b\n")
	print(x \%*\% b)
	cat("original y\n")
	print(y)
}
