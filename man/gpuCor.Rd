\name{gpuCor}
\alias{gpuCor}
\title{Calculate Various Correlation Coefficients With a GPU}

\description{
	The correlation coefficient will be calculated
	for each pair $x_i$, $y_j$ where $x_i$ is a column of $x$ and $y_j$ is a
	column of $y$.  Currently, Pearson's and Kendall's correlation
	coefficient are implemented.  Pearson's may be calculated for
	data sets containing NAs in which case, the implementation behaves
	as R-native cor function with use="pairwise.complete".
}

\usage{
	gpuCor(x, y = NULL, method = c("pearson", "kendall"), anyNAs = FALSE) 
}

\arguments{
	\item{x}{a matrix of floating point values in which each column is a 
		random variable.}  
	\item{y}{a matrix of floating point values in which each column is a 
		random variable.}
	\item{method}{a string.  Either "pearson" or "kendall".}
	\item{anyNAs}{a logical value.  Set to TRUE to do Pearson's with NAs in 
		your data set.}
}

\value{
	For method "pearson" and anyNAs=TRUE, a list with matrices pairs,
	coefficients, and ts.  The matrix entry $i$, $j$ for pairs represents
	the number of pairs of entries $x_i^k$, $y_j^k$ (the $k$-th entry from
	$x_i$ and $y_j$ respectively) where neither are NA.  These are the
	number of entries actually used to calculate the coefficients.
	Entry $i$, $j$ of the coefficients matrix is the correlation coeffcient
	for $x_i$, $y_j$.  Entry $i$, $j$ of the ts matrix is the t-score of the
	$i$, $j$ entry of the coefficient matrix.

	For method "pearson" and anyNAs=FALSE, a matrix of floating point
	numbers where entry $i$, $j$ is the correlation coeffcient for $x_i$, $y_j$.

	For method "kendall", a matrix of floating point numbers where entry 
	$i$, $j$ is the correlation coeffcient for $x_i$, $y_j$.
}

\seealso{
	cor
}

\examples{
numAvars <- 5
numBvars <- 10
numSamples <- 30
A <- matrix(runif(numAvars*numSamples), numSamples, numAvars)
B <- matrix(runif(numBvars*numSamples), numSamples, numBvars)
gpuCor(A, B, "pearson")
gpuCor(A, B, "kendall")
A[3,2] <- NA
gpuCor(A, B, "pearson", TRUE)
}
