#include<stdio.h>
#include<math.h>
#include<string.h>

#include<cublas.h>

#include<cuseful.h>

void grammSchmidtProcess(size_t rows, size_t cols, float * matA) {
	size_t
		i, j;
	float 
		invnorm, sum;

	for(j = 0; j < cols; j++) {
		for(i = 0; i < j; i++) {
			sum = cublasSdot(rows, matA+rows*i, 1, matA+rows*j, 1);
			cublasSaxpy(rows, -sum, matA+rows*i, 1, matA+rows*j, 1);
		}
		sum = cublasSdot(rows, matA+rows*j, 1, matA+rows*j, 1);
		invnorm = 1.f / sqrtf(sum);
		cublasSscal(rows, invnorm, matA+rows*j, 1);
	}
}

// matA has order rows x cols
// matQ has order rows x cols
// matR has order cols x cols 
// please allocate space for matQ and matR before calling qrDecomp
void qrDecomp(size_t rows, size_t cols, const float * matA, float * matQ, 
	float * matR) 
{
	cudaMemcpy(matQ, matA, rows*cols*sizeof(float), cudaMemcpyDeviceToDevice);
	grammSchmidtProcess(rows, cols, matQ);

	cublasSgemm('T', 'N', cols, cols, rows, 1.f, matQ, rows, matA, rows, 
		0.f, matR, cols);
}

// matX has order rows x cols
// vectY has length rows
// vectB has length cols
// please allocate space for vectB before calling qrlsSolver
void qrlsSolver(size_t rows, size_t cols, const float * matX, 
	const float * vectY, float * vectB)
{
	size_t fbytes = sizeof(float);
	float 
		* matQ, * matR;

	cudaMalloc((void **)&matQ, rows*cols*fbytes);
	cudaMalloc((void **)&matR, cols*cols*fbytes);
	checkCudaError("qrlsSolver : attempted gpu memory allocation");

	qrDecomp(rows, cols, matX, matQ, matR); 

	// compute the vector Q^t * Y
	// vectQtY[i] = dotProduct(Q's col i, Y)
	cublasSgemv('T', rows, cols, 1.f, matQ, rows, vectY, 1, 0.f, vectB, 1);
	cublasStrsv('U', 'N', 'N', cols, matR, cols, vectB, 1);
}
