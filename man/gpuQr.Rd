\name{gpuQr}
\alias{gpuQr}
\title{Estimate the QR decomposition for a matrix}

\description{
	This function estimates the QR decomposition for a matrix using
	column pivoting and householder matricies.  The work is done on a
	GPU.
}

\usage{
gpuQr(a)
}

\arguments{
	\item{a}{a matrix of floating point numbers.  This is the matrix that will
	be decomposed into Q and R factors.}
}

\value{
a list consisting of the following elements:  Q, R, pivot, rank. Q is a matrix
representing the Q factor of input a.  R is a matrix representing the
upper triangular R factor of input a.  The pivot element is a vector
representing the final permutation of the columns of a after the QR
decomposition function finishes decomposing a.  The format of pivot is similar
to the base R function 'qr'.  The rank element is a single integer representing
an estimation of the rank of input matrix 'a' based on the results of the
QR decomposition.  In some rare cases, this rank can be wildly different from
the actual rank of the matrix a.  It is only an estimation.
}

\references{
Bjorck, Ake (1996) \emph{Numerical methods for least squares problems}. SIAM.
}

\examples{
	# get some random data of any shape at all
	x <- matrix(runif(1000), 50, 20)
	qr <- gpuQr(x)
	print(qr)
}
