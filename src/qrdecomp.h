void qrdecompMGS(int rows, int cols, float * da, float * dq, float * dr, 
	int * pivots);
void getQRDecomp(int rows, int cols, float * dq, float * da, int * pivot);
void qrSolver(int rows, int cols, float * matX, float * vectY, float * vectB);
void getQRDecompPacked(int rows, int cols, float tol, float * dQR,
	int * pivot, float * qrAux, int * rank);

void getInverseFromQR(int rows, int cols, const float * dQ, const float * dR, 
	float * dInverse);
void solveFromQR(int rows, int cols, const float * matQ, const float * matR,
	const float * vectY, float * vectB);
