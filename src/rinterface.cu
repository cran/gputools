#include<stdio.h>
#include<string.h>
#include<correlation.h>
#include<distance.h>
#include<granger.h>
#include<hcluster.h>
#include<lr.h>
#include<matmult.h>
#include<rinterface.h>

void rpmcc(const float * samplesA, const int * numSamplesA,
	const float * samplesB, const int * numSamplesB, const int * sampleSize,
	float * numPairs, float * correlations, float * signifs)
{
	pmcc(samplesA, *numSamplesA, samplesB, *numSamplesB, *sampleSize,
		numPairs, correlations, signifs);
}

void rformatInput(const int * images, 
	const int * xcoords, const int * ycoords, const int * zcoords,
	const int * mins, const int * maxes,
	const float * evs, const int * numrows, const int * numimages, 
	float * output)
{
	getData(images, xcoords, ycoords, zcoords, mins, maxes, evs, 
		*numrows, *numimages, output);
}

void rformatOutput(const int * imageList1, const int * numImages1, 
	const int * imageList2, const int * numImages2, 
	const int * structureid,
	const double * cutCorrelation, const int * cutPairs,
	const double * correlations, const double * signifs, const int * numPairs, 
	double * results, int * nrows)
{
	*nrows = (int) parseResults(imageList1, *numImages1, imageList2, 
		*numImages2, *structureid, *cutCorrelation, *cutPairs, 
		correlations, signifs, numPairs, results);
}

void rsetDevice(const int * device) {
	setDevice(*device);
}

void rgetDevice(int * device) {
	getDevice(device);
}

void rtestT(const float * pairs, const float * coeffs, const int * n, 
	float * ts) 
{
	testSignif(pairs, coeffs, (size_t) *n, ts);
}

void rhostT(const float * pairs, const float * coeffs, const int * n, 
	float * ts) 
{
	hostSignif(pairs, coeffs, (size_t) *n, ts);
}

void rSignifFilter(const double * data, int * rows, double * results) {
	*rows = signifFilter(data, (size_t) *rows, results);
}

void gSignifFilter(const float * data, int * rows, float * results) {
	*rows = gpuSignifFilter(data, (size_t) *rows, results);
}

void RcublasPMCC(const float * samplesA, const int * numSamplesA,
	const float * samplesB, const int * numSamplesB, const int * sampleSize,
	float * correlations)
{
	cublasPMCC(samplesA, *numSamplesA, samplesB, *numSamplesB, *sampleSize, 
		correlations);
}

void RhostKendall(const float * X, const float * Y, const int * n, 
	double * answer)
{
	*answer = hostKendall(X, Y, *n);
}

void RpermHostKendall(const float * X, const int * nx, const float * Y, 
	const int * ny, const int * sampleSize, double * answers)
{
	permHostKendall(X, *nx, Y, *ny, *sampleSize, answers);
}

void RgpuKendall(const float * X, const int * nx, const float * Y, 
	const int * ny, const int * sampleSize, double * answers)
{
	masterKendall(X, *nx, Y, *ny, *sampleSize, answers);
}

void dlr(const int * numParams, const int * numObs, const float * obs,
	float * outcomes, float * coeffs, const float * epsilon, 
	const float * ridge, const int * maxiter) {

	logRegression(*numParams, *numObs, obs, outcomes, coeffs, *epsilon, 
		*ridge, *maxiter);
}

void RGranger(const int * rows, const int * cols, const float * y, 
	const int * p, float * results) {

	gpuGrangerTest(*rows, *cols, y, *p, results);
}

dist_method getDistEnum(const char * methodStr)
{
	if(0 == strcmp(methodStr,"maximum"))	return MAXIMUM;
	if(0 == strcmp(methodStr,"manhattan"))	return MANHATTAN;
	if(0 == strcmp(methodStr,"canberra"))	return CANBERRA;
	if(0 == strcmp(methodStr,"binary"))		return BINARY;
	if(0 == strcmp(methodStr,"minkowski"))	return MINKOWSKI;
//	if(0 == strcmp(methodStr,"dot"))		return DOT;
	return EUCLIDEAN;
}

hc_method getClusterEnum(const char * methodStr)
{
	if(0 == strcmp(methodStr,"complete"))		return COMPLETE;
	if(0 == strcmp(methodStr,"wpgma"))			return WPGMA;
	if(0 == strcmp(methodStr,"average"))		return AVERAGE;
	if(0 == strcmp(methodStr,"median"))			return MEDIAN;
	if(0 == strcmp(methodStr,"centroid"))		return CENTROID;
	if(0 == strcmp(methodStr,"flexible_group"))	return FLEXIBLE_GROUP;
	if(0 == strcmp(methodStr,"flexible"))		return FLEXIBLE;
	if(0 == strcmp(methodStr,"ward"))			return WARD;
	if(0 == strcmp(methodStr,"mcquitty"))		return MCQUITTY;
	return SINGLE;
}

void Rdistclust(const char ** distmethod, const char ** clustmethod, 
	const float * points, const int * numPoints, const int * dim,
	int * merge, int * order, float * val)
{
	dist_method dmeth = getDistEnum(*distmethod); 
	hc_method hcmeth = getClusterEnum(*clustmethod); 

	size_t dpitch = 0;
	float * gpuDistances = NULL;

	distanceLeaveOnGpu(dmeth, 2.f, points, *dim, *numPoints, 
		&gpuDistances, &dpitch);

	size_t len = (*numPoints) - 1;
	float 
		lambda = 0.5f, beta = 0.5f;
	int 
		* presub, * presup;

	presub = (int *) malloc(len*sizeof(int));
	presup = (int *) malloc(len*sizeof(int));

	hclusterPreparedDistances(gpuDistances, dpitch, *numPoints, 
		presub, presup, val, hcmeth, lambda, beta);

	formatClustering(len, presub, presup, merge, order);

	free(presub);
	free(presup);
}

void Rdistances(const float * points, const int * numPoints, const int * dim,
	float * distances, const char ** method, const float *p)
{
	dist_method nummethod = getDistEnum(*method); 

	distance(points, (*dim)*sizeof(float), *numPoints, points, 
		(*dim)*sizeof(float), *numPoints, *dim, distances, 
		(*numPoints)*sizeof(float), nummethod, *p);
}

void Rhcluster(const float * distMat, const int * numPoints, 
	int * merge, int * order, float * val, const char ** method)
{
	hc_method nummethod = getClusterEnum(*method); 
	
	size_t len = (*numPoints) - 1;
	size_t pitch = (*numPoints) * sizeof(float);
	float lambda = 0.5;
	float beta = 0.5;
	int 
		* presub, * presup;

	presub = (int *) malloc(len*sizeof(int));
	presup = (int *) malloc(len*sizeof(int));

	hcluster(distMat, pitch, *numPoints, presub, presup, val, nummethod,
		lambda, beta);

	formatClustering(len, presub, presup, merge, order);

	free(presub);
	free(presup);
}

void formatClustering(const int len, const int * sub,  const int * sup, 
	int * merge, int * order)
{
	for(size_t i = 0; i < len; i++) {
		merge[i] = -(sub[i] + 1);
		merge[i+len] = -(sup[i] + 1);
	}

	for(size_t i = 0; i < len; i++) {
		for(size_t j = i+1; j < len; j++) {
			if((merge[j] == merge[i]) || (merge[j] == merge[i+len]))
				merge[j] = i + 1;
			if((merge[j+len] == merge[i]) || (merge[j+len] == merge[i+len]))
				merge[j+len] = i + 1;
			if(((merge[j+len] < 0) && (merge[j] > 0)) 
				|| ((merge[j] > 0) && (merge[j+len] > 0) 
				&& (merge[j] > merge[j+len]))) {
				int holder = merge[j];
				merge[j] = merge[j+len];
				merge[j+len] = holder; 
			}
		}
	}
	getPrintOrder(len, merge, order);
}

void getPrintOrder(const int len, const int * merge, int * order)
{
	int 
		level = len-1, otop = len;

	depthFirst(len, merge, level, &otop, order);
}

void depthFirst(const int len, const int * merge, int level, int * otop, 
	int * order)
{
	int
		left = level, right = level + len;

	if(merge[right] < 0) {
		order[*otop] = -merge[right];
		(*otop)--;
	} else
		depthFirst(len, merge, merge[right]-1, otop, order);

	if(merge[left] < 0) {
		order[*otop] = -merge[left];
		(*otop)--;
	} else
		depthFirst(len, merge, merge[left]-1, otop, order);
}

void RgpuMatMult(float * a, int * rowsa, int * colsa, 
	float * b, int * rowsb, int * colsb, float * result) {

	gpuMatMult(a, *rowsa, *colsa, b, *rowsb, *colsb, result);
}
