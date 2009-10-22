#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<R.h>
#include<cuseful.h>
#include<mi.h>

#define NTHREADS 16

__device__ float get_min(int n, const float * x)
{
	float min = x[0];
	for(int i = 1; i < n; i++) {
		if(x[i] < min)
			min = x[i];
	}
	return min;
}

__device__ float get_max(int n, const float * x)
{
	float max = x[0];
	for(int i = 1; i < n; i++) {
		if(x[i] > max)
			max = x[i];
	}
	return max;
}

__device__ void scale(float knot_max, int n, float * x)
{
	float
		max = get_max(n, x),
		min = get_min(n, x),
		delta = max - min;

	for(int i = 0; i < n; i++)
		x[i] = (knot_max * (x[i] - min)) / delta;
}

// bins must be initialized to zero before calling get_bin_scores
__device__ void get_bin_scores(int nbins, int order, float * knots, float x,
	float * bins)
{
	int i0 = (int)floorf(x) + order - 1;
	if(i0 >= nbins)
		i0 = nbins - 1;

	bins[i0] = 1.f;
	for(int i = 2; i <= order; i++) {
		for(int j = i0 - i + 1; j <= i0; j++) {
			int
				dpom1 = j + i - 1, dpo = j + i, dp1 = j + 1;
			float
				ld = 0.f, rd = 0.f;

			if(knots[dpom1] != knots[j])
				ld = (x - knots[j]) / (knots[dpom1] - knots[j]);
			if(knots[dpo] != knots[dp1])
				rd = (knots[dpo] - x) / (knots[dpo] - knots[dp1]);

			bins[j] = ld * bins[j] + rd * bins[j + 1];
		}
	}
}

__device__ float get_entropy(int nbins, int nsamples, float * bin_scores)
{
	float entropy = 0.f, prob, logp;
	for(int i = 0; i < nbins; i++) {
		prob = 0.f;
		for(int j = 0; j < nsamples; j++)
			prob += bin_scores[j * nbins + i];
		prob /= (float) nsamples;

		if(prob <= 0.f)
			logp = 0.f;
		else
			logp = __log2f(prob);

		entropy += prob * logp;
	}
	return -entropy;
}

__device__ float get_joint_entropy(int nbins, int nsamples, float * x_bins,
	float * y_bins)
{
	float prob, logp, xy_entropy;
	xy_entropy = 0.f;
	for(int i = 0; i < nbins; i++) {
		for(int j = 0; j < nbins; j++) {
			prob = 0.f;
			for(int k = 0; k < nsamples; k++)
				prob += x_bins[k * nbins + i] * y_bins[k * nbins + j];
			prob /= (float)nsamples;

			if(prob <= 0.f)
				logp = 0.f;
			else
				logp = log2(prob);

			xy_entropy += prob * logp;
		}
	}
	return -xy_entropy;
}

__global__ void gmi(int nbins, int order, int nknots, float * knots,
	int nsamples, int nx, float * x, int pitch_x, float * bins_x, int pitch_bx,
	int ny, float * y, int pitch_y,float * bins_y, int pitch_by,
	float * mi, int pitch_mi)
{
	int
		col_x = blockDim.x * blockIdx.x + threadIdx.x,
		col_y = blockDim.y * blockIdx.y + threadIdx.y;

	if((col_x >= nx) || (col_y >= ny))
		return;

	float
		knot_max = knots[nknots - 1],
		* cur_x = x + col_x * pitch_x,
		* cur_bins_x = bins_x + col_x * pitch_bx,
		entropy_x,
		* cur_y = y + col_y * pitch_y,
		* cur_bins_y = bins_y + col_y * pitch_by,
		entropy_y,
		entropy_xy;

	scale(knot_max, nsamples, cur_x);
	scale(knot_max, nsamples, cur_y);

	for(int i = 0; i < nsamples; i++) {
		get_bin_scores(nbins, order, knots, cur_x[i], cur_bins_x + i * nbins);
		get_bin_scores(nbins, order, knots, cur_y[i], cur_bins_y + i * nbins);
	}
	entropy_x = get_entropy(nbins, nsamples, cur_bins_x);
	entropy_y = get_entropy(nbins, nsamples, cur_bins_y);
	entropy_xy = get_joint_entropy(nbins, nsamples, cur_bins_x, cur_bins_y);
	(mi + col_y * pitch_mi)[col_x] = entropy_x + entropy_y - entropy_xy;
}

int initKnots(int nbins, int order, float ** knots)
{
	int
		om1 = order - 1,
		degree = nbins - 1,
		dpo = degree + order,
		nknots = dpo + 1;

	*knots = Calloc(nknots, float);	
	for(int i = 0; i < nknots; i++) {
		if(i <= om1)
			(*knots)[i] = 0.0;
		else if(i <= degree)
			(*knots)[i] = (*knots)[i-1] + 1.0;
		else
			(*knots)[i] = (*knots)[degree] + 1.0;
	}
	return nknots;
}

void bSplineMutualInfo(int nbins, int order, int nsamples, int nx,
	const float * x, int ny, const float * y, float * out_mi)
{
	size_t
		pitch_x, pitch_y,
		pitch_bins_x, pitch_bins_y,
		pitch_mi;
	int
		nknots, nblocks_x, nblocks_y;
	float
		* knots, * dknots, 
		* dx, * dy,
		* dbins_x, * dbins_y,
		* dmi;

	cudaMallocPitch((void **)&dx, &pitch_x, nsamples * sizeof(float), nx);
	cudaMemcpy2D(dx, pitch_x, x, nsamples * sizeof(float),
		nsamples * sizeof(float), nx, cudaMemcpyHostToDevice);

	cudaMallocPitch((void **)&dbins_x, &pitch_bins_x,
		nbins * nsamples * sizeof(float), nx);
	cudaMemset2D(dbins_x, pitch_bins_x, 0, nbins * nsamples * sizeof(float),
		nx);

	cudaMallocPitch((void **)&dy, &pitch_y, nsamples * sizeof(float), ny);
	cudaMemcpy2D(dy, pitch_y, y, nsamples * sizeof(float),
		nsamples * sizeof(float), ny, cudaMemcpyHostToDevice);
	
	cudaMallocPitch((void **)&dbins_y, &pitch_bins_y,
		nbins * nsamples * sizeof(float), ny);
	cudaMemset2D(dbins_y, pitch_bins_y, 0, nbins * nsamples * sizeof(float),
		ny);

	nknots = initKnots(nbins, order, &knots);
	cudaMalloc((void **)&dknots, nknots * sizeof(float));
	cudaMemcpy(dknots, knots, nknots * sizeof(float), cudaMemcpyHostToDevice);
	Free(knots);

	cudaMallocPitch((void **)&dmi, &pitch_mi, nx * sizeof(float), ny);

	checkCudaError("bSplineMutualInfoSingle: calculating entropy on gpu");

	nblocks_x = nx / NTHREADS;
	if(nblocks_x * NTHREADS < nx)
		nblocks_x++;

	nblocks_y = ny / NTHREADS;
	if(nblocks_y * NTHREADS < ny)
		nblocks_y++;

	dim3
		gridDim(nblocks_x, nblocks_y), blockDim(NTHREADS, NTHREADS);

	int
		px = pitch_x / sizeof(float), pbx = pitch_bins_x / sizeof(float),
		py = pitch_y / sizeof(float), pby = pitch_bins_y / sizeof(float),
		pm = pitch_mi / sizeof(float);

	gmi<<<gridDim, blockDim>>>(nbins, order, nknots, dknots, nsamples,
		nx, dx, px, dbins_x, pbx, ny, dy, py, dbins_y, pby, dmi, pm);

	checkCudaError("bSplineMutualInfoSingle: calculating entropy on gpu");

	cudaFree(dknots);
	cudaFree(dbins_x);
	cudaFree(dx);
	cudaFree(dbins_y);
	cudaFree(dy);

	cudaMemcpy2D(out_mi, nx * sizeof(float), dmi, pitch_mi,
		nx * sizeof(float), ny, cudaMemcpyDeviceToHost);

	checkCudaError("bSplineMutualInfoSingle: calculating entropy on gpu");
	cudaFree(dmi);
}
