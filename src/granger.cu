#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<cuseful.h>
#include<granger.h>

#define max(a, b) ((a > b)?a:b)

#define THREADSPERDIM	16

// mX has order rows x cols
// vectY has length rows
__global__ void getRestricted(int count, int rows, int cols, 
	float * mX, int mXdim, float * vY, int vYdim, float * mQ, int mQdim,
	float * mR, int mRdim, float * vectB, int vectBdim) {

	int 
		m = blockIdx.x * THREADSPERDIM + threadIdx.x, n,
		i, j, k;
	float 
		sum, invnorm,
		* X, * Y, * Q, * R, * B,
		* coli, * colj, 
		* colQ, * colX;

	if(m >= count) return;
	if(m == 1) n = 0;
	else n = 1;

	X = mX + (m * mXdim);
	// initialize the intercepts
	for(i = 0; i < rows; i++)
		X[i] = 1.f;

	Y = vY + (m * count + n) * vYdim;
	B = vectB + m * vectBdim;
	Q = mQ + m * mQdim;
	R = mR + m * mRdim;

	// initialize Q with X ...
	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++)
			Q[i+j*rows] = X[i+j*rows];
	}

	// gramm-schmidt process to find Q
	for(j = 0; j < cols; j++) {
		colj = Q+rows*j;
		for(i = 0; i < j; i++) {
			coli = Q+rows*i;
			sum = 0.f;
			for(k = 0; k < rows; k++)
				sum += coli[k] * colj[k];
			for(k = 0; k < rows; k++)
				colj[k] -= sum * coli[k];
		}
		sum = 0.f;
		for(i = 0; i < rows; i++)
			sum += colj[i] * colj[i];
		invnorm = 1.f / sqrtf(sum);
		for(i = 0; i < rows; i++)
			colj[i] *= invnorm;
	}
	for(i = cols-1; i > -1; i--) {
		colQ = Q+i*rows;
		// matmult Q * X -> R
		for(j = 0; j < cols; j++) {
			colX = X+j*rows;
			sum = 0.f;
			for(k = 0; k < rows; k++)
				sum += colQ[k] * colX[k];
			R[i+j*cols] = sum;
		}
		sum = 0.f;
		// compute the vector Q^t * Y -> B
		for(j = 0; j < rows; j++) 
			sum += colQ[j] * Y[j];
		// back substitution to find the x for Rx = B
		for(j = cols-1; j > i; j--)
			sum -= R[i+j*cols] * B[j];

		B[i] = sum / R[i+i*cols];
	}
}

// mX has order rows x cols
// vectY has length rows
__global__ void getUnrestricted(int count, int rows, int cols, 
	float * mX, int mXdim, float * vY, int vYdim, float * mQ, int mQdim,
	float * mR, int mRdim, float * vectB, int vectBdim) {

	int 
		m = blockIdx.x * THREADSPERDIM + threadIdx.x, 
		n = blockIdx.y * THREADSPERDIM + threadIdx.y, 
		i, j, k;
	float 
		sum, invnorm,
		* X, * Y, * Q, * R, * B,
		* coli, * colj, 
		* colQ, * colX;
	if((m >= count) || (n >= count) || (m == n)) return;

	X = mX + (m * count + n) * mXdim;
	// initialize the intercepts
	for(i = 0; i < rows; i++) 
		X[i] = 1.f;

	Y = vY + (m*count+n) * vYdim;
	B = vectB + (m*count+n) * vectBdim;
	Q = mQ + (m*count+n) * mQdim;
	R = mR + (m*count+n) * mRdim;

	// initialize Q with X ...
	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++)
			Q[i+j*rows] = X[i+j*rows];
	}

	// gramm-schmidt process to find Q
	for(j = 0; j < cols; j++) {
		colj = Q+rows*j;
		for(i = 0; i < j; i++) {
			coli = Q+rows*i;
			sum = 0.f;
			for(k = 0; k < rows; k++)
				sum += coli[k] * colj[k];
			for(k = 0; k < rows; k++)
				colj[k] -= sum * coli[k];
		}
		sum = 0.f;
		for(i = 0; i < rows; i++)
			sum += colj[i] * colj[i];
		invnorm = 1.f / sqrtf(sum);
		for(i = 0; i < rows; i++)
			colj[i] *= invnorm;
	}
	for(i = cols-1; i > -1; i--) {
		colQ = Q+i*rows;
		// matmult Q * X -> R
		for(j = 0; j < cols; j++) {
			colX = X+j*rows;
			sum = 0.f;
			for(k = 0; k < rows; k++)
				sum += colQ[k] * colX[k];
			R[i+j*cols] = sum;
		}
		sum = 0.f;
		// compute the vector Q^t * Y -> B
		for(j = 0; j < rows; j++) 
			sum += colQ[j] * Y[j];
		// back substitution to find the x for Rx = B
		for(j = cols-1; j > i; j--)
			sum -= R[i+j*cols] * B[j];

		B[i] = sum / R[i+i*cols];
	}
}

__global__ void ftest(int p, int rows, int cols, int rCols, int unrCols, 
	float * obs, int obsDim, float * rCoeffs, int rCoeffsDim, 
	float * unrCoeffs, int unrCoeffsDim, float * rdata, int rdataDim,
	float * unrdata, int unrdataDim, float * results) {

	int 
		i = blockIdx.x * THREADSPERDIM + threadIdx.x, 
		j = blockIdx.y * THREADSPERDIM + threadIdx.y, 
		k, m;
	float 
		kobs, fp = (float) p, frows = (float) rows,
		rSsq, unrSsq,
		rEst, unrEst,
		score = 0.f, 
		* tObs, * tRCoeffs, * tUnrCoeffs, 
		* tRdata, * tUnrdata; 

	if((i >= cols) || (j >= cols)) return;
	if(i == j) return;

	tObs = obs + i*cols*obsDim+j*obsDim;

	tRCoeffs = rCoeffs + i*rCoeffsDim;
	tRdata = rdata + i*rdataDim;
	
	tUnrCoeffs = unrCoeffs + i*cols*unrCoeffsDim+j*unrCoeffsDim;
	tUnrdata = unrdata + i*cols*unrdataDim+j*unrdataDim;

	rSsq = unrSsq = 0.f;
	for(k = 0; k < rows; k++) {
		unrEst = rEst = 0.f;
		kobs = tObs[k];
		for(m = 0; m < rCols; m++)
			rEst += tRCoeffs[m] * tRdata[k+m*rows];
		for(m = 0; m < unrCols; m++) 
			unrEst += tUnrCoeffs[m] * tUnrdata[k+m*rows];
		rSsq   += (kobs - rEst) * (kobs - rEst);
		unrSsq += (kobs - unrEst) * (kobs - unrEst);

	}
	score = ((rSsq - unrSsq)*(frows-2.f*fp-1.f)) / (fp*unrSsq);
	int idx = (i*(cols-1) + ((j<i)? j:(j-1)))*2;
	// printf("i : %d, j : %d, idx : %d\n", i, j, idx);
	results[idx] = score;
	
	float 
		x = score, mfact, alpha, sum = 0.f,
		d1 = (float)p, d2 = (float)rows - 2.f * (float)p - 1.f,
		v = ((float) (d1*x)) / ((float) (d1*x+d2)), 
		a = ((int)d1) / 2, b = ((int)d2) / 2;

	if((int)d1 % 2 != 0) a++;
	if((int)d2 % 2 != 0) b++; 
	if((a < 0) || (b < 0)) return; 

	alpha = a+b-1;
	for(k = a+b-2; (k > 1) && (k > b-2); k--) 
		alpha *= k;

	mfact = a;
	for(m = 2; m < a; m++)
		mfact *= (float)m;

	for(k = -1, m = a; m < a+b; m++, k++) {
		sum += (alpha*__powf(v, (float)m)*__powf(1.f-v, (float)(a+b-1-m))) 
			/ mfact;
		if(b+k <= 0) alpha = 1.f;
		else alpha /= (float)b+k;
		mfact *= m+1;
	}
	results[idx+1] = 1.f - sum;
}

void gpuGrangerTest(int rows, int cols, const float * y, int p, 
	float * results) {

	if(cols < 2)
		fatal("The Granger test needs at least 2 variables.\n");
	int
		i, j, k, t = p+1,
		fbytes = sizeof(float),
		embedRows = rows-p, embedCols = t*2;
	float 
		* Y, * rQ, * rR,
		* unrQ, * unrR,
		* restricted, * unrestricted,
		* rdata, * unrdata,
		* dresults;
	size_t 
		size = cols*cols*fbytes, partSize = embedRows*size;

	cudaMalloc((void **)&Y, embedCols*partSize);

	cudaMalloc((void **)&rQ, t*embedRows*cols*fbytes);
	cudaMalloc((void **)&rR, t*t*cols*fbytes);
	cudaMalloc((void **)&rdata, t*embedRows*cols*fbytes);
	cudaMalloc((void **)&restricted, t*cols*fbytes);

	cudaMalloc((void **)&unrQ, (embedCols-1)*partSize);
	cudaMalloc((void **)&unrR, (embedCols-1)*(embedCols-1)*size);
	cudaMalloc((void **)&unrestricted, (embedCols-1)*size);
	cudaMalloc((void **)&unrdata, (embedCols-1)*partSize);

	size_t resultSize = 2*cols*(cols-1)*fbytes;
	cudaMalloc((void **)&dresults, resultSize);
	checkCudaError("granger : attemped gpu memory allocation");

	// results = (float *)xmalloc(resultSize);

	int
		Ydim =  embedCols * embedRows,
		rQdim = t * embedRows, rRdim = t * t,
		rdataDim = t*embedRows, restrictedDim = t,
		unrQdim = (embedCols-1) * embedRows, 
		unrRdim = (embedCols-1) * (embedCols-1),
		unrestrictedDim = embedCols-1, unrdataDim = (embedCols-1)*embedRows;
	float 
		* ypos, * rdataPos, * unrdataPos;

	int 
		skip = 2*embedRows, colBytes = embedRows*fbytes;
	const float * vectA, * vectB;
	float * evenCols;

	for(i = 0; i < cols; i++) {
		rdataPos = rdata+i*rdataDim;
		evenCols = rdataPos+embedRows;
		vectA = y+i*rows; 
		for(j = 0; j < cols; j++) {
			if(i == j) continue;
			
			ypos = Y+(i*cols+j)*Ydim;
			unrdataPos = unrdata+(i*cols+j)*unrdataDim;

			vectB = y+j*rows;

			for(k = 0; k < p+1; k++) { // produce t subcols
				cudaMemcpy(ypos+k*skip, vectA+(p-k), embedRows*fbytes, 
					cudaMemcpyHostToDevice);
				cudaMemcpy(ypos+k*skip+embedRows, vectB+(p-k), 
					embedRows*fbytes, cudaMemcpyHostToDevice);
			}
			cudaMemcpy(unrdataPos+embedRows, ypos+skip,
				(embedCols-2)*embedRows*fbytes, cudaMemcpyDeviceToDevice);
		}
		// build restricted data from last set of unrestricted data
		// only need one per column, not one for each pairing
		for(k = 0; k < embedCols-2; k+=2) { 
			cudaMemcpy(evenCols+(k*embedRows)/2, unrdataPos+(1+k)*embedRows, 
				colBytes, cudaMemcpyDeviceToDevice);
		}
	}

	int numBlocks;

	numBlocks = cols / THREADSPERDIM;
	if(numBlocks * THREADSPERDIM < cols) numBlocks++;

	dim3 
		dimRGrid(numBlocks), 
		dimRBlock(THREADSPERDIM), 
		dimUnrGrid(numBlocks, numBlocks), 
		dimUnrBlock(THREADSPERDIM, THREADSPERDIM);

	getRestricted<<<dimRGrid, dimRBlock>>>(cols, embedRows, t, rdata, rdataDim,
		Y, Ydim, rQ, rQdim, rR, rRdim, restricted, restrictedDim);
	getUnrestricted<<<dimUnrGrid, dimUnrBlock>>>(cols, embedRows, embedCols-1, 
		unrdata, unrdataDim, Y, Ydim, unrQ, unrQdim, unrR, unrRdim, 
		unrestricted, unrestrictedDim);

	cudaThreadSynchronize();

	ftest<<<dimUnrGrid, dimUnrBlock>>>(p, embedRows, cols, t, embedCols-1, Y, 
		Ydim, restricted, restrictedDim, unrestricted, unrestrictedDim, rdata, 
		rdataDim, unrdata, unrdataDim, dresults); 
	cudaMemcpy(results, dresults, resultSize, cudaMemcpyDeviceToHost);

	cudaFree(Y);
	cudaFree(rQ);
	cudaFree(unrQ);
	cudaFree(rR);
	cudaFree(unrR);
	cudaFree(restricted);
	cudaFree(unrestricted);
	cudaFree(rdata);
	cudaFree(unrdata);
	cudaFree(dresults);
}
