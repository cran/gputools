void bSplineMutualInfo(int cols, int nBins, int splineOrder,
	int rowsA, const float * A, int rowsB, const float * B, 
	float * mutualInfo);

void bSplineMutualInfoSingle(int cols,
	int nBins, int splineOrder, int rows, const float * A,
	float * mutualInfo);
