#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<cublas.h>

#include"cuseful.h"
#include"qrdecomp.h"

#define NTHREADS 512

__global__ void getColNorms(int rows, int cols, float * da, int lda, 
	float * colNorms)
{
	int colIndex = threadIdx.x + blockIdx.x * blockDim.x;
	float 
		norm = 0.f, term, 
		* col;

	if(colIndex >= cols) return;

	col = da+colIndex*lda;

	for(int i = 0; i < rows; i++) {
		term = col[i];
		term *= term;
		norm += term;
	}
	colNorms[colIndex] = sqrtf(norm);
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

	if(start >= n) return;

	int localMaxIndex = start;
	for(int i = start+1; i < end; i++) {
		if(i >= n) break;
		if(data[i] > data[localMaxIndex]) localMaxIndex = i;
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
	if(threadIdx.x == 0)
		*maxIndex = maxIndicies[0];
}

__global__ void gpuSwapCol(int rows, float * dArray, int coli, int * dColj,
	int * dPivot)
{
	int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if(rowIndex >= rows) return;

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

	if(threadIdx.x > rows) return;

	sum = 0.f;
	for(i = threadIdx.x ; i < rows; i += NTHREADS) {
		if((threadIdx.x == 0) && (i == 0)) continue;
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
	ident = (float *) xmalloc(rows*rows*fbytes);
	memset(ident, 0, rows*rows*fbytes);

	for(int i = 0; i < rows; i++)
		ident[i*(1+rows)] = 1.f;

	cublasAlloc(rows*rows, fbytes, (void **)&dIdent);
	cublasSetMatrix(rows, rows, fbytes, ident, rows, dIdent, rows);
	free(ident);

	cublasAlloc(cols, fbytes, (void **)&dColNorms);
	cublasAlloc(cols, fbytes, (void **)&dPivot);
	cublasAlloc(rows, fbytes, (void **)&dv);
	cublasAlloc(rows*rows, fbytes, (void **)&dH);
	cublasAlloc(rows*rows, fbytes, (void **)&dr);
	cublasAlloc(1, fbytes, (void **)&dMaxIndex);

	checkCublasError("getQRDecomp: line 96");

	for(int i = 0; i < cols; i++)
		pivot[i] = i;
	cublasSetVector(cols, sizeof(int), pivot, 1, dPivot, 1);

	for(int k = 0; (k < cols) && (k < rows-1); k++) {
		rowsk = rows - k;
		colsk = cols - k;
		dak = da+(rows+1)*k;
		drk = dr+(rows+1)*k;

		nblocks = colsk / nthreads;
		if(nblocks*nthreads < colsk) nblocks++;

		getColNorms<<<nblocks, nthreads>>>(rowsk, colsk, dak, rows, dColNorms);
		gpuFindMax<<<1, nblocks>>>(colsk, dColNorms, nthreads, dMaxIndex);
		gpuSwapCol<<<nRowBlocks, nthreads>>>(rows, da, k, dMaxIndex, dPivot);

		int work = rowsk / nthreads;
		if(work * nthreads < rowsk) work++;
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
		checkCublasError("getQRDecomp: line 155");
		checkCudaError("getQRDecomp: line 114");
	} // finally, da holds R, dq holds Q

	cublasFree(dIdent);
	cublasFree(dv);
	cublasFree(dH);
	cublasFree(dr);
	cublasFree(dColNorms);
	cublasFree(dMaxIndex);

	cublasGetVector(cols, sizeof(int), dPivot, 1, pivot, 1);
	cublasFree(dPivot);
	checkCublasError("getQRDecomp: line 163");
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
	int 
		fbytes = sizeof(float),
		* pivot;
	float * matQ;

	pivot = (int *) xmalloc(cols*sizeof(int));
	cublasAlloc(rows*rows, fbytes, (void **)&matQ);
	checkCublasError("qrlsSolver : line 165");

	getQRDecomp(rows, cols, matQ, matX, pivot);

	// compute the vector Q^t * Y
	// vectQtY[i] = dotProduct(Q's col i, Y)
	cublasSgemv('N', cols, rows, 1.f, matQ, rows, vectY, 1, 0.f, vectB, 1);
	cublasStrsv('U', 'N', 'N', cols, matX, rows, vectB, 1);
	checkCublasError("qrlsSolver : line 173");

	for(int i = 0; i < cols; i++) {
		if(pivot[i] != i) {
			int j = find(cols, pivot, i);
			cublasSswap(1, vectB+i, 1, vectB+j, 1);
			swap(i, j, pivot);
		}
	}
	checkCublasError("qrlsSolver : line 182");
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

	rowR = (float *) xmalloc(cols*fbytes);

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
	free(rowR);
}
