#include<stdio.h>
#include<math.h>
#include<string.h>

#include<cublas.h>

#include<qrdecomp.h>
#include<cuseful.h>

#define NUMTHREADS 512
#define TRUE  1
#define FALSE !TRUE

__global__ void gpuLogistic(size_t nobs, float * x, const float * y, 
	float * means, float * weights)
{
	float 
		mean, deriv, xi, yi;
	size_t
		row = blockIdx.x*blockDim.x + threadIdx.x; 

	if(row > nobs) return;

	mean = xi = x[row];
	yi = y[row];
//	wi = w[row];

	mean = 1.f / (1.f + __expf(-mean));
	deriv = mean*(1.f-mean);

	means[row] = mean;
	weights[row] = deriv;
	__syncthreads();
	x[row] = (deriv * xi + (yi - mean));
}

__host__ void logistic(size_t nobs, float * gpuX, const float * gpuY, 
	float * gpuMeans, float * gpuWeights)
{
	size_t 
		numBlocks = nobs / NUMTHREADS;

	if(numBlocks*NUMTHREADS < nobs) numBlocks++;

	dim3
		dimGrid(numBlocks), dimBlock(NUMTHREADS);

	gpuLogistic<<<dimGrid, dimBlock>>>(nobs, gpuX, gpuY, gpuMeans, gpuWeights);
}

__global__ void gpuWeights(size_t numParams, size_t numObs, const float * obs, 
	const float * weights, float * holder)
{
	size_t
		i,
		row = blockIdx.x*blockDim.x + threadIdx.x; 

	if(row > numObs) return;
	for(i = 0; i < numParams; i++)
		holder[row+i*numObs] = obs[row+i*numObs] * weights[row];
}

__host__ void applyWeights(size_t numParams, size_t numObs, const float * obs, 
	const float * weights, float * holder)
{
	size_t 
		numBlocks = numObs / NUMTHREADS;

	if(numBlocks*NUMTHREADS < numObs) numBlocks++;

	dim3
		dimGrid(numBlocks), dimBlock(NUMTHREADS);

	gpuWeights<<<dimGrid, dimBlock>>>(numParams, numObs, obs, weights, holder);
}

__global__ void gpuRidge(size_t n, float ridge, float * mat) {
	size_t
		i = blockIdx.x*blockDim.x + threadIdx.x; 

	if(i > n) return;
	mat[i*n+i] += ridge;
}

__host__ void doRidge(size_t n, float ridge, float * mat) {
	size_t 
		numBlocks = n / NUMTHREADS;

	if(numBlocks*NUMTHREADS < n) numBlocks++;

	dim3
		dimGrid(numBlocks), dimBlock(NUMTHREADS);

	gpuRidge<<<dimGrid, dimBlock>>>(n, ridge, mat);
}

__global__ void gpuAbsSub(size_t n, const float * a, float * b) {
	size_t
		i = blockIdx.x*blockDim.x + threadIdx.x; 

	if(i > n) return;
	b[i] = fabsf(a[i] - b[i]);
}

__host__ void absSub(size_t n, const float * a, float * b) {
	size_t 
		numBlocks = n / NUMTHREADS;

	if(numBlocks*NUMTHREADS < n) numBlocks++;

	dim3
		dimGrid(numBlocks), dimBlock(NUMTHREADS);

	gpuAbsSub<<<dimGrid, dimBlock>>>(n, a, b);
}

void transpose(size_t n, float * a) {
	size_t i, j;
	float holder;

	for(i = 0; i < n; i++) {
		for(j = 0; j < i; j++) {
			holder = a[j*n+i];
			a[j*n+i] = a[i*n+j];
			a[i*n+j] = holder;
		}
	}
}

// suggestions for the parameters:
// ridge = 0.000001
// maxiter = 200
// epsilon = 0.0000000001
void logRegression(size_t numParams, size_t numObs, const float * obs, 
	float * outcomes, float * coeffs, float epsilon, float ridge, 
	size_t maxiter)
{
	size_t 
		fbytes = sizeof(float), 
		outBytes = numObs*fbytes, obsBytes = numParams*outBytes, 
		coeffBytes = numParams*fbytes,
		i;
	float 
		sum, cutoff = numObs*numObs*epsilon*epsilon,
		* dObs, * dCoeffs, * dOutcomes,
		* holder, * A,
		* adjy,
		* weights, * oldexpy, * expy;

	cudaMalloc((void**)&dOutcomes, outBytes);
	cudaMalloc((void**)&dObs, obsBytes);
	cudaMalloc((void**)&dCoeffs, coeffBytes);

	cudaMemcpy(dOutcomes, outcomes, outBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dObs, obs, obsBytes, cudaMemcpyHostToDevice);
	checkCudaError("logRegression : attempted gpu memory allocation");
	cudaMemset(dCoeffs, 0, coeffBytes);

	cudaMalloc((void**)&holder, numParams*outBytes);
	cudaMalloc((void**)&A, numParams*coeffBytes);

	cudaMalloc((void**)&adjy, outBytes);
	cudaMalloc((void**)&expy, outBytes);
	cudaMalloc((void**)&oldexpy, outBytes);
	cudaMalloc((void**)&weights, outBytes);
	checkCudaError("logRegression : attempted gpu memory allocation");

	// init(numObs, w, oldexpy);
	float * initExpy = (float *)xmalloc(numObs*fbytes);
	for(i = 0; i < numObs; i++) initExpy[i] = -1.f;
	cudaMemcpy(oldexpy, initExpy, outBytes, cudaMemcpyHostToDevice);
	free(initExpy);
	
	int didConverge = FALSE;
	for(i = 0; i < maxiter; i++) {
		// adjy = obs * coeffs
		cublasSgemv('N', numObs, numParams, 1.f, dObs, numObs, dCoeffs, 1,
			0.f, adjy, 1);

		logistic(numObs, adjy, dOutcomes, expy, weights);

		// A = obs^t * weights * obs + diag(ridge)
		applyWeights(numParams, numObs, dObs, weights, holder);
		cublasSgemm('T', 'N', numParams, numParams, numObs, 1.f, holder, 
			numObs, dObs, numObs, 0.f, A, numParams); 
		doRidge(numParams, ridge, A);
		
		// v = obs^t * wadjy
		// adjy was reused to hold wadjy
		// reusing weights to hold v
		cublasSgemv('T', numObs, numParams, 1.f, dObs, numObs, adjy, 1, 
			0.f, weights, 1);

		qrSolver(numParams, numParams, A, weights, dCoeffs);

		cublasSaxpy(numObs, -1.f, expy, 1, oldexpy, 1);
		sum = cublasSdot(numObs, oldexpy, 1, oldexpy, 1);
		if(sum < cutoff) {
			didConverge = TRUE;
			break;
		}
		cudaMemcpy(oldexpy, expy, outBytes, cudaMemcpyDeviceToDevice);
	}
	if(!didConverge) 
		fprintf(stderr, "The iterative regression algo did not converge to a solution.\n");

	cudaMemcpy(coeffs, dCoeffs, coeffBytes, cudaMemcpyDeviceToHost);

	cudaFree(dCoeffs);
	cudaFree(holder);
	cudaFree(A);
	cudaFree(adjy);
	cudaFree(expy);
	cudaFree(oldexpy);
	cudaFree(weights);
}
