\name{gpuDist}
\alias{gpuDist}
\title{Compute Distances Between Vectors on a GPU}

\description{
	This function computes the distance between each vector of the
	'points' argument using the metric specified by 'method'.
}

\usage{
	gpuDist(points, method = "euclidean", p = 2.0)
}

\arguments{
	\item{points}{a matrix of floating point numbers in which each row is a
		vector in $R^n$ space where $n$ is ncol(points).}
	\item{method}{a string representing the name of the metric to use to
		calculate the distance between the vectors of 'points'.  Currently
		supported values are: "binary", "canberra", "euclidean", "manhattan",
		"maximum", and "minkowski".}
	\item{p}{a floating point parameter for the Minkowski metric.}
}

\value{
	a matrix of floating point numbers whose i, j entry represents the 
	distance between vector i and vector j from the 'points' argument.
}

\seealso{
	dist
}

\examples{
numVectors <- 5
dimension <- 10
Vectors <- matrix(runif(numVectors*dimension), numVectors, dimension)
gpuDist(Vectors, "euclidean")
gpuDist(Vectors, "maximum")
gpuDist(Vectors, "manhattan")
gpuDist(Vectors, "minkowski", 4)
}

\keyword{math}
