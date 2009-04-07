\name{gpuGranger}
\alias{gpuGranger}
\title{Perform Granger Causality Tests for Vectors on a GPU}

\description{
	This function performs, with the aid of a GPU, Granger Causality Tests on 
	permutations of pairs of columns of the input matrix 'varCols'.
}

\usage{
	gpuGranger(varCols, lag)
}

\arguments{
	\item{varCols}{a matrix of floating point values.  Each column represents
		a sequence of observations for a single random variable.}
	\item{lag}{a positive integer by which to offset the sequence of 
		observations to calculate the coefficient for Granger causality.}
}

\value{a two column matrix of floating point values.  The first column is
	F-statistic from the Granger causality test.  The second column is a
	rough estimation of the p-value for the F-statistic of the same row.
	Note that the p-values for F-statistics greater than magnitude one differ
	from that calculated by the package MSBVAR's 'granger.test'.

	To find the Granger causality F-statistic estimating the answer to
	"Does variable j Granger-cause variable i?", look at entry 
	$(i-1)*n - 1) + k$, where $k = j$ if $j < i$ and $k = j-1$ if
	$j > i$ where $n$ is nrow(varCols).
}

\examples{
numRandVars <- 5
numSamples <- 20
randVarSequences <- matrix(runif(numRandVars*numSamples), numSamples, 
	numRandVars)
gpuGranger(randVarSequences, 5)
}
