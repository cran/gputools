#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>

#include<cuseful.h>

#include<mi.h>

#define MAXBINS 32
#define NTHREADS 16

void initKnots(int nBins, int splineOrder, float * knots)
{
	float
		knot,
		max = (float)nBins - (float)splineOrder + 1.f;

	for(int i = 0; i < nBins + splineOrder + 1; i++) {
		if(i < splineOrder)
			knot = 0.f;
		else if(i < nBins)
			knot = (float)i - (float)splineOrder + 1.f;
		else
			knot = max;

		knots[i] = knot / max;
	}
}

__global__ void getMi(int nBins, int splineOrder, const float * knots,
	int cols,
	int nRowsA, const float * pointsA, int pitchA,
	int nRowsB, const float * pointsB, int pitchB,
	float * mi, int pitchMi)
{
	int
		rowA = blockDim.x * blockIdx.x + threadIdx.x,
		rowB = blockDim.y * blockIdx.y + threadIdx.y;
	const float
		* pointRowA, * pointRowB;
	float
		tempKnot,
		denom,
		termA, termB,
		points[2], factor,
		bSplines[2][MAXBINS],
		binSums[2][MAXBINS],
		jointBinSums[MAXBINS][MAXBINS],
		entropyA, entropyB, jointEntropy;

	if((rowA >= nRowsA) || (rowB >= nRowsB) || (nBins > MAXBINS))
		return;

	pointRowA = (const float *) ((char *)pointsA + rowA * pitchA);
	pointRowB = (const float *) ((char *)pointsB + rowB * pitchB);

	for(int i = 0; i < nBins; i++) {
		binSums[0][i] = 0.f;
		binSums[1][i] = 0.f;
		for(int j = 0; j < nBins; j++)
			jointBinSums[i][j] = 0.f;
	}

	for(int x = 0; x < cols; x++) {
		points[0] = pointRowA[x];
		points[1] = pointRowB[x];

		#pragma UNROLL
		for(int i = 0; i < 2; i++) {
			for(int j = 0; j < nBins; j++) {
				if((knots[j] <= points[i]) && (points[i] < knots[j + 1]))
					bSplines[i][j] = 1.f;
				else
					bSplines[i][j] = 0.f;
			}
			
			for(int k = 1; k < splineOrder; k++) {
				for(int j = 0; j < nBins; j++) {
					tempKnot = knots[j];
					denom = knots[j + k] - tempKnot;
					if(denom != 0.f) {
						termA = bSplines[i][j];
						termA *= points[i] - knots[j];
						termA /= denom;
					} else
						termA = 0.f;
			
					tempKnot = knots[j + k + 1];
					denom = tempKnot - knots[j+1];
					if((denom != 0.f) && (j + 1 < nBins)) {
						termB = bSplines[i][j+1];
						termB *= points[i] - tempKnot;
						termB /= denom;
					} else
						termB = 0.f;
			
					bSplines[i][j] = termA - termB;
				}
			}
		}

		for(int i = 0; i < nBins; i++) {
			binSums[0][i] += bSplines[0][i];
			binSums[1][i] += bSplines[1][i];
			for(int j = 0; j < nBins; j++)
				jointBinSums[i][j] += bSplines[0][i] * bSplines[1][j];
		}
	}

	entropyA = entropyB = jointEntropy = 0.f;
	jointEntropy = 0.f;
	for(int i = 0; i < nBins; i++) {
		factor = binSums[0][i] / (float)cols;
		if(factor > 0.f)
			entropyA += factor * __logf(factor);

		factor = binSums[1][i] / (float)cols;
		if(factor > 0.f)
			entropyB += factor * __logf(factor);

		for(int j = 0; j < nBins; j++) {
			factor = jointBinSums[i][j] / (float)cols;
			if(factor > 0.f)
				jointEntropy += factor * __logf(factor);
		}
	}

	float * miRow = (float *)((char *)mi + rowA * pitchMi);
	miRow[rowB] = -entropyA - entropyB + jointEntropy;
}

void bSplineMutualInfo(int cols, int nBins, int splineOrder,
	int rowsA, const float * A, int rowsB, const float * B, 
	float * mutualInfo)
{
	size_t
		pitchA, pitchB, pitchMi;
	int
		nBlocksA, nBlocksB;
	float
		* knots,
		* dA, * dB,
		* dKnots, * dMi;

	knots = (float *) xmalloc((nBins + splineOrder + 1) * sizeof(float));
	initKnots(nBins, splineOrder, knots);

	cudaMalloc((void **)&dKnots, (nBins + splineOrder + 1) * sizeof(float));
	cudaMallocPitch((void **)&dA, &pitchA, cols * sizeof(float), rowsA);
	cudaMallocPitch((void **)&dB, &pitchB, cols * sizeof(float), rowsB);
	checkCudaError("bSplineMutualInfoSingle: line 272");
	cudaMallocPitch((void **)&dMi, &pitchMi, rowsB * sizeof(float), rowsA);
	checkCudaError("bSplineMutualInfoSingle: line 274");

	cudaMemcpy(dKnots, knots, (nBins + splineOrder + 1) * sizeof(float),
		cudaMemcpyHostToDevice);

	free(knots);
	
	cudaMemcpy2D(dA, pitchA, A, cols * sizeof(float),
		cols * sizeof(float), rowsA, cudaMemcpyHostToDevice);
	cudaMemcpy2D(dB, pitchB, B, cols * sizeof(float),
		cols * sizeof(float), rowsB, cudaMemcpyHostToDevice);
	
	nBlocksA = rowsA / NTHREADS;
	if(nBlocksA * NTHREADS < rowsA)
		nBlocksA++;

	nBlocksB = rowsB / NTHREADS;
	if(nBlocksB * NTHREADS < rowsB)
		nBlocksB++;

	dim3
		gridDim(nBlocksA, nBlocksB), blockDim(NTHREADS, NTHREADS);

	getMi<<<gridDim, blockDim>>>(nBins, splineOrder, dKnots, cols,
		rowsA, dA, pitchA, rowsB, dB, pitchB, dMi, pitchMi);
	checkCudaError("bSplineMutualInfoSingle: line 291");

	cudaFree(dKnots);
	cudaFree(dA);
	cudaFree(dB);

	cudaMemcpy2D(mutualInfo, rowsB * sizeof(float), dMi, pitchMi,
		rowsB * sizeof(float), rowsA, cudaMemcpyDeviceToHost);
	checkCudaError("bSplineMutualInfoSingle: line 298");

	cudaFree(dMi);
}

void bSplineMutualInfoSingle(int cols,
	int nBins, int splineOrder, int rows, const float * A,
	float * mutualInfo)
{
	size_t
		pitchA, pitchMi;
	int nBlocks;
	float
		* knots,
		* dA, * dKnots, * dMi;

	knots = (float *) xmalloc((nBins + splineOrder + 1) * sizeof(float));
	initKnots(nBins, splineOrder, knots);

	cudaMalloc((void **)&dKnots, (nBins + splineOrder + 1) * sizeof(float));
	cudaMallocPitch((void **)&dA, &pitchA, cols * sizeof(float), rows);
	checkCudaError("bSplineMutualInfoSingle: line 272");
	cudaMallocPitch((void **)&dMi, &pitchMi, rows * sizeof(float), rows);
	checkCudaError("bSplineMutualInfoSingle: line 274");

	cudaMemcpy(dKnots, knots, (nBins + splineOrder + 1) * sizeof(float),
		cudaMemcpyHostToDevice);
	free(knots);
	cudaMemcpy2D(dA, pitchA, A, cols * sizeof(float),
		cols * sizeof(float), rows, cudaMemcpyHostToDevice);
	
	nBlocks = rows / NTHREADS;
	if(nBlocks * NTHREADS < rows)
		nBlocks++;

	dim3
		gridDim(nBlocks, nBlocks), blockDim(NTHREADS, NTHREADS);

	getMi<<<gridDim, blockDim>>>(nBins, splineOrder, dKnots, cols,
		rows, dA, pitchA, rows, dA, pitchA, dMi, pitchMi);
	checkCudaError("bSplineMutualInfoSingle: line 291");

	cudaFree(dKnots);
	cudaFree(dA);

	cudaMemcpy2D(mutualInfo, rows * sizeof(float), dMi, pitchMi,
		rows * sizeof(float), rows, cudaMemcpyDeviceToHost);
	checkCudaError("bSplineMutualInfoSingle: line 298");

	cudaFree(dMi);
}
