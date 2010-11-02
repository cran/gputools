#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<cublas.h>
#include<cuseful.h>
#include<R.h>
#include<matmult.h>

void gpuMatMult(int tpA, int tpB, float * a, int rowsa, int colsa, 
	float * b, int rowsb, int colsb, float * c)
{
	float
		* gpua, * gpub, * gpuc;

	cublasInit();
	checkCublasError("gpuMatMult device initialization");

	cublasAlloc(rowsa*colsa, sizeof(float), (void **) &gpua);
	checkCublasError("gpuMatMult memory allocation");
	cublasAlloc(rowsb*colsb, sizeof(float), (void **) &gpub);
	checkCublasError("gpuMatMult memory allocation");
	char opA = tpA ? 'T' : 'N';
	char opB = tpB ? 'T' : 'N';
	int rowsOpA = tpA ? colsa : rowsa;
	int colsOpA = tpA ? rowsa : colsa;
	int colsOpB = tpB ? rowsb : colsb;
	cublasAlloc(rowsOpA*colsOpB, sizeof(float), (void **) &gpuc);
	checkCublasError("gpuMatMult memory allocation");

	cublasSetVector(rowsa*colsa, sizeof(float), a, 1, gpua, 1);
	cublasSetVector(rowsb*colsb, sizeof(float), b, 1, gpub, 1);

	cublasSgemm(opA, opB, rowsOpA, colsOpB, colsOpA, 1.0, gpua, rowsa, gpub, rowsb,	0.0, gpuc, rowsOpA);

	cublasGetVector(rowsOpA*colsOpB, sizeof(float), gpuc, 1, c, 1);
	checkCublasError("gpuMatMult read from gpu memory");

	cublasFree(gpua);
	cublasFree(gpub);
	cublasFree(gpuc);
	cublasShutdown();
}

void gpu16MatMult(float * a, int rowsa, int colsa, 
	float * b, int rowsb, int colsb, float * c)
{
	if(colsa != rowsb) {
		fprintf(stderr, "error: matrix dimensions mismatched for matrix multiplication\n"); 
		return;
	}
	float
		* bigA, * bigB, * bigC,
		* gpua, * gpub, * gpuc;
	int
		bigRowsA, bigColsA,	// we are gonna blow these up to
		bigRowsB, bigColsB; // the nearest power of 16 to help cuda

	bigRowsA = ((rowsa >> 4) + ((rowsa & 15)? 1:0)) << 4;
	bigColsA = ((colsa >> 4) + ((colsa & 15)? 1:0)) << 4;
	bigRowsB = ((rowsb >> 4) + ((rowsb & 15)? 1:0)) << 4;
	bigColsB = ((colsb >> 4) + ((colsb & 15)? 1:0)) << 4;

	cublasInit();
	checkCublasError("gpu16MatMult device initialization");

	cublasAlloc(bigRowsA*bigColsA, sizeof(float), (void **) &gpua);
	cublasAlloc(bigRowsB*bigColsB, sizeof(float), (void **) &gpub);
	cublasAlloc(bigRowsA*bigColsB, sizeof(float), (void **) &gpuc);
	checkCublasError("gpu16MatMult memory allocation");

	bigA = Calloc(bigRowsA*bigColsA, float);
	for(int i = 0; i < colsa; i++)
		memcpy(&bigA[i*bigRowsA], &a[i*rowsa], rowsa*sizeof(float));

	bigB = Calloc(bigRowsB*bigColsB, float);
	for(int i = 0; i < colsb; i++)
		memcpy(&bigB[i*bigRowsB], &b[i*rowsb], rowsb*sizeof(float));

	cublasSetVector(bigRowsA*bigColsA, sizeof(float), bigA, 1, gpua, 1);
	cublasSetVector(bigRowsB*bigColsB, sizeof(float), bigB, 1, gpub, 1);
	checkCublasError("gpu16MatMult write to gpu memory");

	cublasSgemm('N', 'N', bigRowsA, bigColsB, bigColsA, 1.0, gpua, bigRowsA,
		gpub, bigRowsB, 0.0, gpuc, bigRowsA);
	checkCublasError("gpu16MatMult gpu routine execution");

	bigC = Calloc(bigRowsA*bigColsB, float);
	cublasGetVector(bigRowsA*bigColsB, sizeof(float), gpuc, 1, bigC, 1);
	checkCublasError("gpu16MatMult read from gpu memory");

	for(int i = 0; i < colsb; i++)
		memcpy(&c[i*rowsa], &bigC[i*bigRowsA], rowsa*sizeof(float));

	cublasFree(gpua);
	cublasFree(gpub);
	cublasFree(gpuc);
	cublasShutdown();
	Free(bigA);
	Free(bigB);
	Free(bigC);
}
