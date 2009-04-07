#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<cublas.h>

#include"cuseful.h"

#define HALF RAND_MAX/2

int fatal(const char * msg) {
	if(msg != NULL) fputs(msg, stderr);
	return 0;
	// exit(EXIT_FAILURE);
}

void * xmalloc(size_t nbytes) {
    register void * result = malloc(nbytes);
    if(result == 0) fatal("Failed allocating more RAM; maybe out of RAM.");
    return result;
}

float * fMalloc(size_t numElts)
{
	float * result;

	result = NULL;
	result = (float *) malloc(numElts*sizeof(float));
	if(result == NULL) fatal("error allocating host memory\n");
	return result;
}

double * dMalloc(size_t numElts)
{
	double * result;

	result = NULL;
	result = (double *) malloc(numElts*sizeof(double));
	if(result == NULL) fatal("error allocating host memory\n");
	return result;
}

size_t * stMalloc(size_t n) {
	size_t * vect = NULL;
	vect = (size_t *)malloc(n*sizeof(size_t));
	if(vect == NULL) fatal("error allocating host memory\n");
	return vect;
}

float * getMatFromFile(int rows, int cols, const char * fn) {
	FILE * matFile;
	matFile = fopen(fn, "r");
	if(matFile == NULL) {
		size_t length = 32+strlen(fn);
		char line[length];
		sprintf(line, "unable to open file %s", fn);
		fatal(line);
	}
	float * mat = (float *)xmalloc(rows*cols*sizeof(float));
	int i, j, err;
	for(i = 0; i < rows; i++) {
		for(j = 0; j < cols; j++) {
			err = fscanf(matFile, " %f ", mat+i+j*rows);
			if(err == EOF) {
				size_t length = 32+strlen(fn);
				char line[length];
				sprintf(line, "file %s incorrect: formatting or size", fn);
				fatal(line);
			}
		}
		fscanf(matFile, " \n ");
	}
	fclose(matFile);
	return mat;
}

char * getTime() {
	time_t curtime;
	struct tm *loctime;
	curtime = time(NULL);
	loctime = localtime(&curtime);
	
	return asctime(loctime);
}

void printVect(int n, const float * vect, const char * msg) {
	if(msg != NULL) puts(msg);
	for(int i = 0; i < n; i++) {
		printf("%6.4f, ", vect[i]);
		if((i+1)%10 == 0) printf("\n");
	}
	if(n%10 != 0) printf("\n");
	if(msg != NULL) puts("----------");
}

void printMat(int rows, int cols, const float * mat, const char * msg) {
	int i;
	if(msg != NULL) puts(msg);
	for(i = 0; i < rows; i++)
		printVect(cols, mat+i*cols, NULL);
	if(msg != NULL) puts("----------");
}

void getRandVect(float * vect, size_t n) {
	srand(time(0));
	for(size_t i = 0; i < n; i++)
		vect[i] = ((float)rand())/((float)RAND_MAX);
}

void getBernVect(float * vect, size_t n) {
	srand(time(0));
	for(size_t i = 0; i < n; i++) {
		if(rand() <= HALF) vect[i] = 1.f;
		else vect[i] = 0.f;
	}
}

void checkCudaError(const char * msg) {
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err) {
		fprintf(stderr, "cuda error : %s : %s\n", msg, cudaGetErrorString(err));
		fatal(NULL);
	}
}

char * cublasGetErrorString(cublasStatus err)
{
	switch(err) {
		case CUBLAS_STATUS_SUCCESS :
			return "operation completed successfully";
		case CUBLAS_STATUS_NOT_INITIALIZED :
			return "CUBLAS library not initialized";
		case CUBLAS_STATUS_ALLOC_FAILED :
			return "resource allocation failed";
		case CUBLAS_STATUS_INVALID_VALUE :
			return "unsupported numerical value was passed to function";
		case CUBLAS_STATUS_ARCH_MISMATCH :
			return "function requires an architectural feature absent from \
			the architecture of the device";
		case CUBLAS_STATUS_MAPPING_ERROR :
			return "access to GPU memory space failed";
		case CUBLAS_STATUS_EXECUTION_FAILED :
			return "GPU program failed to execute";
		case CUBLAS_STATUS_INTERNAL_ERROR :
			return "an internal CUBLAS operation failed";
		default :
			return "unknown error type";
	}
}

void checkCublasError(const char * msg)
{
	cublasStatus err = cublasGetError();
	if(err != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "cublas error : %s : %s\n", msg, 
			cublasGetErrorString(err));
		fatal(NULL);
	}
}
