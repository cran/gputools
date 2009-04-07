#include<stdio.h>
#include<string.h>
#include"nocuda.h"

int noCudaFatal() 
{
	fputs("CUDA not installed.  Please uninstall the gputools library, install CUDA, and then reinstall the gputools library.\n\n", stderr);
	return 0;
	// exit(EXIT_FAILURE);
}

void rpmcc(const float * samplesA, const int * numSamplesA,
	const float * samplesB, const int * numSamplesB, const int * sampleSize,
	float * numPairs, float * correlations, float * signifs)
{
	noCudaFatal();
}

void rformatInput(const int * images, 
	const int * xcoords, const int * ycoords, const int * zcoords,
	const int * mins, const int * maxes,
	const float * evs, const int * numrows, const int * numimages, 
	float * output)
{
	noCudaFatal();
}

void rformatOutput(const int * imageList1, const int * numImages1, 
	const int * imageList2, const int * numImages2, 
	const int * structureid,
	const double * cutCorrelation, const int * cutPairs,
	const double * correlations, const double * signifs, const int * numPairs, 
	double * results, int * nrows)
{
	noCudaFatal();
}

void rsetDevice(const int * device)
{
	noCudaFatal();
}

void rgetDevice(int * device)
{
	noCudaFatal();
}

void rtestT(const float * pairs, const float * coeffs, const int * n, 
	float * ts) 
{
	noCudaFatal();
}

void rhostT(const float * pairs, const float * coeffs, const int * n, 
	float * ts) 
{
	noCudaFatal();
}

void rSignifFilter(const double * data, int * rows, double * results)
{
	noCudaFatal();
}

void gSignifFilter(const float * data, int * rows, float * results)
{
	noCudaFatal();
}

void RcublasPMCC(const float * samplesA, const int * numSamplesA,
	const float * samplesB, const int * numSamplesB, const int * sampleSize,
	float * correlations)
{
	noCudaFatal();
}

void RhostKendall(const float * X, const float * Y, const int * n, 
	double * answer)
{
	noCudaFatal();
}

void RpermHostKendall(const float * X, const int * nx, const float * Y, 
	const int * ny, const int * sampleSize, double * answers)
{
	noCudaFatal();
}

void RgpuKendall(const float * X, const int * nx, const float * Y, 
	const int * ny, const int * sampleSize, double * answers)
{
	noCudaFatal();
}

void dlr(const int * numParams, const int * numObs, const float * obs,
	float * outcomes, float * coeffs, const float * epsilon, 
	const float * ridge, const int * maxiter)
{
	noCudaFatal();
}

void RGranger(const int * rows, const int * cols, const float * y, 
	const int * p, float * results)
{
	noCudaFatal();
}

void Rdistclust(const char ** distmethod, const char ** clustmethod, 
	const float * points, const int * numPoints, const int * dim,
	int * merge, int * order, float * val)
{
	noCudaFatal();
}

void Rdistances(const float * points, const int * numPoints, const int * dim,
	float * distances, const char ** method, const float *p)
{
	noCudaFatal();
}

void Rhcluster(const float * distMat, const int * numPoints, 
	int * merge, int * order, float * val, const char ** method)
{
	noCudaFatal();
}

void RgpuMatMult(float * a, int * rowsa, int * colsa, 
	float * b, int * rowsb, int * colsb, float * result)
{

	noCudaFatal();
}
