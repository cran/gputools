#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<cublas.h>
#include<R.h>

#include"cuseful.h"
#include"qrdecomp.h"

#define NTHREADS 512

__global__ void getColNorms(int rows, int cols, float * da, int lda, 
	float * colNorms)
{
	int colIndex = threadIdx.x + blockIdx.x * blockDim.x;
	float 
		sum = 0.f, term,
		* col;

	if(colIndex >= cols)
		return;

	col = da + colIndex * lda;

	// debug printing
	// printf("printing column %d\n", colIndex);
	// for(int i = 0; i < rows; i++)
	// printf("%f, ", col[i]);
	// puts("");
	// end debug printing

	for(int i = 0; i < rows; i++) {
		term = col[i];
		term *= term;
		sum += term;
	}

	// debug printing
	// printf("norm %f\n", norm);
	// end debug printing

	colNorms[colIndex] = sum;
}

__global__ void gpuFindMax(int n, float * data, int threadWorkLoad, 
	int * maxIndex)
{
	int
		j, k,
		start = threadWorkLoad * threadIdx.x,
		end = start + threadWorkLoad;
	__shared__ int maxIndicies[NTHREADS];

	maxIndicies[threadIdx.x] = -1;

	if(start >= n)
		return;

	int localMaxIndex = start;
	for(int i = start+1; i < end; i++) {
		if(i >= n)
			break;
		if(data[i] > data[localMaxIndex])
			localMaxIndex = i;
	}
	maxIndicies[threadIdx.x] = localMaxIndex;
	__syncthreads();

	for(int i = blockDim.x >> 1; i > 0; i >>= 1) {
		if(threadIdx.x < i) {
			j = maxIndicies[threadIdx.x];
			k = maxIndicies[i + threadIdx.x];
			if((j != -1) && (k != -1) && (data[j] < data[k])) 
				maxIndicies[threadIdx.x] = k;
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		*maxIndex = maxIndicies[0];
		// debug printing
		// printf("max index: %d\n", *maxIndex);
		// printf("max norm: %f\n", data[*maxIndex]);
		// end debug printing
	}
}

__global__ void gpuSwapCol(int rows, float * dArray, int coli, int * dColj,
	int * dPivot)
{
	int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if(rowIndex >= rows)
		return;

	int colj = coli + (*dColj);
	float fholder;

	fholder = dArray[rowIndex+coli*rows];
	dArray[rowIndex+coli*rows] = dArray[rowIndex+colj*rows];
	dArray[rowIndex+colj*rows] = fholder;

	if((blockIdx.x == 0) && (threadIdx.x == 0)) {
		int iholder = dPivot[coli];
		dPivot[coli] = dPivot[colj];
		dPivot[colj] = iholder;
	}
}

__global__ void makeHVector(int rows, float * input, float * output)
{
	int
		i, j;
	float 
		elt, sum;
	__shared__ float 
		beta, sums[NTHREADS];

	if(threadIdx.x >= rows)
		return;

	sum = 0.f;
	for(i = threadIdx.x ; i < rows; i += NTHREADS) {
		if((threadIdx.x == 0) && (i == 0))
			continue;
		elt = input[i];
		output[i] = elt;
		sum += elt * elt;
	}
	sums[threadIdx.x] = sum;
	__syncthreads();
	
	for(i = blockDim.x >> 1; i > 0 ; i >>= 1) {
		j = i+threadIdx.x;
		if((threadIdx.x < i) && (j < rows)) 
			sums[threadIdx.x] += sums[j];
		__syncthreads();
	}

	if(threadIdx.x == 0) {
		elt = input[0];
		float norm = sqrtf(elt * elt + sums[0]);

		if(elt > 0) 
			elt += norm;
		else 
			elt -= norm;

		output[0] = elt;

		norm = elt * elt + sums[0];
		beta = sqrtf(2.f / norm);
	}
	__syncthreads();

	for(i = threadIdx.x; i < rows; i += NTHREADS)
		output[i] *= beta;
}

int findMax(int n, const float * data)
{
	int maxIdx = 0;
	for(int i = 1; i < n; i++)
		if(data[i] > data[maxIdx]) maxIdx = i;
	return maxIdx;
}

void swap(int i, int j, int * array)
{
	int oldValue = array[i];
	array[i] = array[j];
	array[j] = oldValue;
}

// use householder xfrms and column pivoting to get the R factor of the
// QR decomp of matrix da:  Q*A*P=R, equiv A*P = Q^t * R
// Q is stored on the gpu in dq, please pre-allocate device memory
// R is stored on the gpu in da, destroying the contents of da
// pivot stores the permutation of the cols of da, pre-allocate host mem
//		using pivot, you can recover P, or at least mimic it's action
__host__ void getQRDecomp(int rows, int cols, float * dq, float * da, 
	int * pivot)
{
	int
		nblocks, nthreads = NTHREADS,
		nRowBlocks = rows / nthreads,
		fbytes = sizeof(float),
		rowsk, colsk,
		* dMaxIndex, * dPivot;
	float
		// elt, alpha,
		* ident, * dColNorms,
		*dv, * dH, * dIdent,
		* dr, * drk, * dak;

	if(nRowBlocks*nthreads < rows) nRowBlocks++;

	// the ident is needed as a term for the householder xfrm
	ident = Calloc(rows * rows, float);
	memset(ident, 0, rows*rows*fbytes);

	for(int i = 0; i < rows; i++)
		ident[i*(1+rows)] = 1.f;

	cublasAlloc(rows*rows, fbytes, (void **)&dIdent);
	cublasSetMatrix(rows, rows, fbytes, ident, rows, dIdent, rows);
	Free(ident);

	cublasAlloc(cols, fbytes, (void **)&dColNorms);
	cublasAlloc(cols, fbytes, (void **)&dPivot);
	cublasAlloc(rows, fbytes, (void **)&dv);
	cublasAlloc(rows*rows, fbytes, (void **)&dH);
	cublasAlloc(rows*rows, fbytes, (void **)&dr);
	cublasAlloc(1, fbytes, (void **)&dMaxIndex);

	checkCublasError("getQRDecomp:");

	for(int i = 0; i < cols; i++)
		pivot[i] = i;
	cublasSetVector(cols, sizeof(int), pivot, 1, dPivot, 1);

	for(int k = 0; (k < cols) && (k < rows-1); k++) {
		rowsk = rows - k;
		colsk = cols - k;
		dak = da+(rows+1)*k;
		drk = dr+(rows+1)*k;

		nblocks = colsk / nthreads;
		if(nblocks*nthreads < colsk)
			nblocks++;

		getColNorms<<<nblocks, nthreads>>>(rowsk, colsk, dak, rows, dColNorms);
		gpuFindMax<<<1, nblocks>>>(colsk, dColNorms, nthreads, dMaxIndex);
		gpuSwapCol<<<nRowBlocks, nthreads>>>(rows, da, k, dMaxIndex, dPivot);

		makeHVector<<<1, nthreads>>>(rowsk, dak, dv); 

		// dH will hold I - beta*v*v^t
		cublasScopy(rows*rows, dIdent, 1, dH, 1);
		cublasSger(rowsk, rowsk, -1.f, dv, 1, dv, 1, dH+k*rows+k, rows);

		// A = dH*A
		cublasScopy(rows*colsk, da+k*rows, 1, dr+k*rows, 1);
		cublasSsymm('L', 'U', rowsk, colsk, 1.f, dH+k*rows+k, rows, drk, rows, 
	 		0.f, dak, rows);

		// Q = dH * Q
		if(k == 0) {
	 		cublasScopy(rows*rows, dH, 1, dq, 1);
		} else {
			cublasScopy(rows*rows, dq, 1, dr, 1);
	 		cublasSsymm('L', 'U', rows, rows, 1.f, dH, rows, dr, rows, 
	 			0.f, dq, rows);
		}
		checkCublasError("getQRDecomp:");
		checkCudaError("getQRDecomp:");
	} // finally, da holds R, dq holds Q

	cublasFree(dIdent);
	cublasFree(dv);
	cublasFree(dH);
	cublasFree(dr);
	cublasFree(dColNorms);
	cublasFree(dMaxIndex);

	cublasGetVector(cols, sizeof(int), dPivot, 1, pivot, 1);
	cublasFree(dPivot);
	checkCublasError("getQRDecomp:");
}

int find(int n, int * array, int toFind)
{
	int retVal = -1;
	for(int i = 0; i < n; i++) {
		if(array[i] == toFind) {
			retVal = i;
			break;
		}
	}
	return retVal;
}

// solves XB=Y for B
// matX has order rows x cols
// vectY has length rows
// vectB has length cols
// please allocate space for vectB before calling qrlsSolver
void qrSolver(int rows, int cols, float * matX, float * vectY, float * vectB)
{
    int * pivot;
    float * matQ;

    pivot = Calloc(cols, int);
    cublasAlloc(rows * rows, sizeof(float), (void **)&matQ);
    checkCublasError("qrSolver");

    getQRDecomp(rows, cols, matQ, matX, pivot);

    // compute the vector Q^t * Y
    // vectQtY[i] = dotProduct(Q's col i, Y)
    cublasSgemv('N', cols, rows, 1.f, matQ, rows, vectY, 1, 0.f, vectB, 1);
    cublasStrsv('U', 'N', 'N', cols, matX, rows, vectB, 1);
    checkCublasError("qrSolver");

    for(int i = 0; i < cols; i++) {
        if(pivot[i] != i) {
            int j = find(cols, pivot, i);
            cublasSswap(1, vectB+i, 1, vectB+j, 1);
            swap(i, j, pivot);
        }
    }
    checkCublasError("qrSolver");
    Free(pivot);
}

// vv a work in progress not ready for primetime
// solves XB=Y for B
// matX has order rows x cols
// vectY has length rows
// vectB has length cols
// please allocate space for vectB before calling qrlsSolver
void qrSolver2(int rows, int cols, float * dX, float * dY, float * dB)
{
	int 
		* pivot,
		rank, maxRank = rows > cols ? cols : rows;
	float
		* qrAux,
		* hostIdent,
		* dQR, * dQ, * drInverse, * dxInverse;

	pivot = Calloc(cols, int);
	qrAux = Calloc(maxRank, float);

	cublasAlloc(rows * cols, sizeof(float), (void **)&dQR);
	cublasAlloc(rows * cols, sizeof(float), (void **)&dQ);
	cublasAlloc(rows * cols, sizeof(float), (void **)&dxInverse);
	checkCublasError("qrSolver2:");

	cublasScopy(rows*cols, dX, 1, dQR, 1);
	checkCublasError("qrSolver2:");

	getQRDecompPacked(rows, cols, 0.00001, dQR, pivot, qrAux, &rank);
	checkCublasError("qrSolver2:");

	hostIdent = (float *) Calloc(cols * cols, float);
	for(int i = 0; i < cols; i++)
		hostIdent[i + i * cols] = 1.f;

	cublasAlloc(cols * cols, sizeof(float), (void **)&drInverse);
	cublasSetMatrix(cols, cols, sizeof(float), hostIdent, cols,
		drInverse, cols);
	checkCublasError("qrSolver2:");
	Free(hostIdent);

	cublasStrsm('L', 'U', 'N', 'N', cols, cols, 1.f, dQR, rows,
		drInverse, cols);
	cublasSgemm('N', 'N', rows, cols, cols, 1.f, dX, rows, drInverse, cols,
		0.f, dQ, rows);
	cublasSgemm('N', 'N', cols, rows, cols, 1.f, drInverse, cols, dQ, rows,
		0.f, dxInverse, cols);
	cublasSgemv('N', cols, rows, 1.f, dxInverse, cols, dY, 1, 0.f, dB, 1);
	checkCublasError("qrSolver2:");

	cublasFree(dQR);
	cublasFree(dQ);
	cublasFree(dxInverse);
	cublasFree(drInverse);

	int j;
	for(int i = 0; i < cols; i++) {
		j = pivot[i];
		if(j != i)
			cublasSswap(1, dB+i, 1, dB+j, 1);
	}
	checkCublasError("qrlsSolver2:");

	Free(pivot);
	Free(qrAux);
}

// finds inverse for X where X has QR decomp X = QR
// dQ has order rows x cols
// dR has order cols x cols
// please allocate space for dInverse before calling getInverseFromQR
// dQ, dR, and dInverse all live on the gpu device
void getInverseFromQR(int rows, int cols, const float * dQ, const float * dR,
	float * dInverse)
{
	float
		* rInverse, * hostIdent;

	if((dQ == NULL) || (dR == NULL) || (dInverse == NULL))
		error("getInverseFromQR: a pointer to a matrix is null");
	if((rows <= 0) || (cols <= 0))
		error("getInverseFromQR: invalid rows or cols argument");

	hostIdent = (float *) Calloc(cols * cols, float);
	for(int i = 0; i < cols; i++)
		hostIdent[i + i * cols] = 1.f;

	cublasAlloc(cols * cols, sizeof(float), (void **)&rInverse);
	cublasSetMatrix(cols, cols, sizeof(float), hostIdent, cols, rInverse, cols);
	checkCublasError("getInverseFromQR:");
	Free(hostIdent);

	cublasStrsm('L', 'U', 'N', 'N', cols, cols, 1.f, dR, cols, rInverse, cols);
	cublasSgemm('N', 'T', cols, rows, cols, 1.f, rInverse, cols, dQ, rows,
		0.f, dInverse, cols);
	checkCublasError("getInverseFromQR:");
	cublasFree(rInverse);
}

// solves XB=Y for B where X has QR decomp X = QR
// matQ has order rows x cols
// matR has order cols x cols
// vectY has length rows
// vectB has length cols
// please allocate space for vectB before calling solveFromQR
void solveFromQR(int rows, int cols, const float * matQ, const float * matR,
	const float * vectY,  float * vectB)
{
	float
		* dQ, * dR,
		* dY, * dB,
		* xInverse;

	if((matQ == NULL) || (matR == NULL) || (vectY == NULL) || (vectB == NULL))
		error("solveFromQR: null array argument");
	if((rows <= 0) || (cols <= 0))
		error("solveFromQR: invalid rows or cols argument");

	cublasAlloc(rows * cols, sizeof(float), (void **)&dQ);
	cublasSetMatrix(rows, cols, sizeof(float), matQ, rows, dQ, rows);
	checkCublasError("solveFromQR:");

	cublasAlloc(cols * cols, sizeof(float), (void **)&dR);
	cublasSetMatrix(cols, cols, sizeof(float), matR, cols, dR, cols);
	checkCublasError("solveFromQR:");

	cublasAlloc(rows * cols, sizeof(float), (void **)&xInverse);
	checkCublasError("solveFromQR:");

	getInverseFromQR(rows, cols, dQ, dR, xInverse);
	cublasFree(dQ);
	cublasFree(dR);

	cublasAlloc(rows, sizeof(float), (void **)&dY);
	cublasSetVector(rows, sizeof(float), vectY, 1, dY, 1);
	checkCublasError("solveFromQR:");

	cublasAlloc(cols, sizeof(float), (void **)&dB);
	checkCublasError("solveFromQR:");

	cublasSgemv('N', cols, rows, 1.f, xInverse, cols, dY, 1, 0.f, dB, 1);
	checkCublasError("solveFromQR:");

	cublasFree(xInverse);
	cublasFree(dY);

	cublasGetVector(cols, sizeof(float), dB, 1, vectB, 1);
	checkCublasError("solveFromQR:");
	cublasFree(dB);
}

// implements the Modified Gram-Schmidt QR decomp
// of the matrix A as found in Numerical methods for least squares
// problems by Ake Bjorck
//
// input:  matrix da in gpu memory
//		the matrix is destroyed by the algorithm
// output:  matrices dq and dr, the QR decomp of A
//		dq has dimension rows x cols, dq is orthonormal
//		dr has dimension cols x cols, dr is the upper triangular Cholesky
//			factor of A^t * A
//		the space for dq and dr should be allocated before calling this
//
void qrdecompMGS(int rows, int cols, float * da, float * dq, float * dr,
	int * pivots)
{
	int
		// pivot, 
		fbytes = sizeof(float);
	float 
		relt, 
		* colQ, * colA,
		* rowR;

	rowR = Calloc(cols, float);

	for(int i = 0; i < cols; i++)
		pivots[i] = i;

	for(int k = 0; k < cols; k++) {
		memset(rowR, 0, cols*fbytes);
		colA = da+k*rows;
/*
		pivot = k + findMaxNormCol(rows, cols-k, colA, &relt);
		if(pivot != k) {
			cublasSswap(rows, colA, 1, da+pivot*rows, 1);
			swap(k, pivot, pivots);
		}
*/
		colQ = dq+k*rows;
		cublasScopy(rows, colA, 1, colQ, 1);
		cublasSscal(rows, 1.f/relt, colQ, 1);
		rowR[k] = relt;
		for(int j = k+1; j < cols; j++) {
			colA = da+j*rows;
			relt = cublasSdot(rows, colQ, 1, colA, 1);
			cublasSaxpy(rows, -relt, colQ, 1, colA, 1);
			rowR[j] = relt;
		}
		cublasSetVector(cols, fbytes, rowR, 1, dr+k, cols);
	}
	Free(rowR);
}

// use householder xfrms and column pivoting to get the R factor of the
// QR decomp of matrix da:  Q*A*P=R, equiv A*P = Q^t * R

__host__ void getQRDecompPacked(int rows, int cols, float tol, float * dQR,
	int * pivot, float * qrAux, int * rank)
{
	int
		nblocks, nthreads = NTHREADS,
		nRowBlocks = rows / nthreads,
		fbytes = sizeof(float),
		rowsk, colsk,
		* dMaxIndex, * dPivot;
	float
		* dColNorms,
		* dV, * dw,
		* zeroes;

	if(nRowBlocks * nthreads < rows)
		nRowBlocks++;

	cublasAlloc(cols, fbytes, (void **) &dColNorms);
	cublasAlloc(cols, fbytes, (void **) &dPivot);
	cublasAlloc(rows*cols, fbytes, (void **) &dV);
	cublasAlloc(rows, fbytes, (void**) &dw);
	cublasAlloc(1, fbytes, (void **) &dMaxIndex);
	checkCublasError("getQRDecompPacked:");

	// Presets the matrix of Householder vectors, dV, to zero.
	// This may be unnecessary, if the cublas calls are adapted
	// to pass the appropriate submatrices.

	zeroes = (float *) Calloc(rows*cols, float);
	cublasSetMatrix(rows, cols, fbytes, zeroes, rows, dV, rows);
	Free(zeroes);

	checkCublasError("getQRDecompPacked:");

	cublasSetVector(cols, sizeof(int), pivot, 1, dPivot, 1);

	int
		rowOffs = 0, k;
	float
		v1,
		minElt; // Minimum tolerable diagonal element

	int maxRank = rows > cols ? cols : rows;
	for(k = 0; k < maxRank; k++, rowOffs += rows) {

		rowsk = rows - k;
		colsk = cols - k;

		nblocks = colsk / nthreads;
		if(nblocks * nthreads < colsk)
			nblocks++;

		getColNorms<<<nblocks, nthreads>>>(rowsk, colsk, dQR + rowOffs, rows,
			dColNorms);
		gpuFindMax<<<1, nblocks>>>(colsk, dColNorms, nthreads, dMaxIndex);
		gpuSwapCol<<<nRowBlocks, nthreads>>>(rows, dQR, k, dMaxIndex, dPivot);

		// Places nonzero elements of V into subdiagonal, although
		// leading element scaled and placed in qrAux.
		//
		cublasScopy(rowsk, dQR + rowOffs + k, 1, dV + rowOffs + k, 1);

		// This should probably be moved to the device.
		//
		// Determines rank, using tolerance to bound condition number.
		// Pivoting has placed the diagonal elements in decreasing order,
		// with the stronger property that the ith. diagonal element has
		// higher magnitude than the 2-norm of the upper i+1st. column
		// (cf. Bjorck).
		//
		// N.B.:  pivoting is not a foolproof way to compute rank, however.
		//
		*rank = k + 1;
		cublasGetVector(1, fbytes, dV + rowOffs + k, 1, &v1, 1);

		if (k == 0) {
			if (abs(v1) < tol) {
				*rank = 0;
				break;
			} else
				minElt = abs((1.f + v1) * tol);
		} else if (abs(v1) < minElt)
			break;

		// Builds Householder vector from maximal column just copied.
		// For now, uses slow memory transfers to modify leading element:
		//      V_1 += sign(V_1) * normV.

		float
			normV = cublasSnrm2(rowsk, dV + rowOffs + k, 1),
			adjust = v1 >= 0 ? normV  : -normV;

		qrAux[k] = 1.f + v1 / adjust;

		v1 += adjust;
		cublasSetVector(1, fbytes, &v1, 1, dV + rowOffs + k, 1);

		// Beta = -2 v^t v

		normV = cublasSdot(rowsk, dV + rowOffs + k, 1, dV + rowOffs +	k, 1);	

		// w = Beta R^t v

		cublasSgemv('T', rows, cols, -2.f / normV, dQR, rows, dV + rowOffs, 1,
			0.f, dw, 1);

		// R = R + v w^t

		cublasSger(rows, cols, 1.0f, dV + rowOffs, 1, dw, 1, dQR, rows);

		// V /= adjust for packed output.

		cublasSscal(rowsk, 1 / adjust, dV + rowOffs + k, 1);

		checkCublasError("getQRDecompPacked:");
		checkCudaError("getQRDecompPacked:");
	}

	// Copies the adjusted lower subdiagonal elements into dQR.

	int offs = 1;
	for (k = 0; k < *rank; k++, offs += (rows + 1)) {
		cublasScopy(rows - k - 1, dV + offs, 1, dQR + offs, 1);
	}

	// dQR now contains the upper-triangular portion of the factorization,
	// R.
	// dV is lower-triangular, and contains the Householder vectors, from
	// which the Q portion can be derived.  An adjusted form of the
	// diagonal is saved in qrAux, while the sub-diagonal portion is
	// written onto QR.

	cublasFree(dV);
	cublasFree(dColNorms);
	cublasFree(dMaxIndex);

	cublasGetVector(cols, sizeof(int), dPivot, 1, pivot, 1);
	cublasFree(dPivot);

	checkCublasError("getQRDecompPacked:");
}
