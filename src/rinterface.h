extern "C" {
	void rpmcc(const float * samplesA, const int * numSamplesA,
		const float * samplesB, const int * numSamplesB, 
		const int * sampleSize, float * numPairs, float * correlations,
		float * signifs);

	void rformatInput(const int * images, 
		const int * xcoords, const int * ycoords, const int * zcoords,
		const int * mins, const int * maxes,
		const float * evs, const int * numrows, const int * numimages, 
		float * output);

	void rformatOutput(const int * imageList1, const int * numImages1, 
		const int * imageList2, const int * numImages2, 
		const int * structureid,
		const double * cutCorrelation, const int * cutPairs,
		const double * correlations, const double * signifs, 
		const int * numPairs, double * results, int * nrows);

	void rsetDevice(const int * device);
	void rgetDevice(int * device);

	void rtestT(const float * pairs, const float * coeffs, const int * n, 
		float * ts);
	void rhostT(const float * pairs, const float * coeffs, const int * n, 
		float * ts); 
	void rSignifFilter(const double * data, int * rows, double * results);
	void gSignifFilter(const float * data, int * rows, float * results);

	void RcublasPMCC(const float * samplesA, const int * numSamplesA,
		const float * samplesB, const int * numSamplesB, 
		const int * sampleSize, float * correlations);

	void RhostKendall(const float * X, const float * Y, const int * n, 
		double * answer);
	void RpermHostKendall(const float * X, const int * nx, const float * Y, 
		const int * ny, const int * sampleSize, double * answers);
	void RgpuKendall(const float * X, const int * nx, const float * Y, 
		const int * ny, const int * sampleSize, double * answers);

	void dlr(const int * numParams, const int * numObs, const float * obs,
		float * outcomes, float * coeffs, const float * epsilon, 
		const float * ridge, const int * maxiter);

	void rgpuGranger(const int * rows, const int * colsy, const float * y, 
		const int * p, float * fStats, float * pValues);
	void rgpuGrangerXY(const int * rows, const int * colsx, const float * x, 
		const int * colsy, const float * y, const int * p, 
		float * fStats, float * pValues);

	void Rdistclust(const char ** distmethod, const char ** clustmethod, 
		const float * points, const int * numPoints, const int * dim,
		int * merge, int * order, float * val);
	void Rdistances(const float * points, const int * numPoints, 
		const int * dim, float * distances, const char ** method,
		const float * p);
	void Rhcluster(const float * distMat, const int * numPoints, 
		int * merge, int * order, float * val, const char ** method);

	void RgpuMatMult(float * a, int * rowsa, int * colsa, 
		float * b, int * rowsb, int * colsb, float * result);

	void R_SVRTrain(float * alpha, float * beta, float * y, float * x,
		float * C, float * kernelwidth, float * eps, int * m, int * n,
		float * StoppingCrit, int * numSvs);
	void R_SVMTrain(float * alpha, float * beta, float * y, float * x,
		float * C, float * kernelwidth, int * m, int * n, float * StoppingCrit,
		int * numSvs, int * numPosSvs);
	void R_produceSupportVectors(int * isRegression, int * m, int * n,
		int * numSVs, int * numPosSVs, const float * x, const float * y,
		const float * alphas, float * svCoefficients, float * supportVectors);
	void R_GPUPredictWrapper(int * m, int * n, int * k, float * kernelwidth,
		const float * Test, const float * Svs, float * alphas,
		float * prediction, float * beta, float * isregression);
	void RgetAucEstimate(int * n, double * classes, double * probs,
		double * outputAuc);

	void RgetQRDecomp(int * rows, int * cols, float * a, float * q, int * pivot,
		int * rank);
	void RqrSolver(int * rows, int * cols, float * matX, float * vectY, 
		float * vectB);

	void rBSplineMutualInfo(int * cols, int * nBins, int * splineOrder,
		int * rowsA, const float * A, int * rowsB, const float * B, 
		float * mutualInfo);
	void rBSplineMutualInfoSingle(int * cols,
		int * nBins, int * splineOrder, int * rows, const float * A,
		float * mutualInfo);
}

void formatClustering(const int len, const int * sub,  const int * sup, 
	int * merge, int * order);
void getPrintOrder(const int len, const int * merge, int * order);
void depthFirst(const int len, const int * merge, int level, int * otop, 
	int * order);
