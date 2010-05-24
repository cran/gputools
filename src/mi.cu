#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<R.h>
#include<cuseful.h>
#include<mi.h>

#define NTHREADS 16

static int initKnots(int nbins, int order, float ** knots)
{
	int
		om1 = order - 1,
		degree = nbins - 1,
		dpo = degree + order,
		nknots = dpo + 1;

	*knots = Calloc(nknots, float);	
	for(int i = 0; i < nknots; i++) {
		if(i <= om1)
			(*knots)[i] = 0.f;
		else if(i <= degree)
			(*knots)[i] = (*knots)[i-1] + 1.f;
		else
			(*knots)[i] = (*knots)[degree] + 1.f;
	}
	return nknots;
}

static float get_min(int n, const float * arr)
{
	float min = arr[0];
	for(int i = 1; i < n; i++) {
		if(arr[i] < min)
			min = arr[i];
	}
	return min;
}

static float get_max(int n, const float * arr)
{
	float max = arr[0];
	for(int i = 1; i < n; i++) {
		if(arr[i] > max)
			max = arr[i];
	}
	return max;
}

__global__ void scale(float knot_max, int nx, float * mins,
	float * maxes, int nsamples, float * x, int pitch_x)
{
	int
		col_idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(col_idx >= nx)
		return;

	float
		min = mins[col_idx], delta = maxes[col_idx] - min,
		* col = x + col_idx * pitch_x;

	for(int i = 0; i < nsamples; i++)
		col[i] = (knot_max * (col[i] - min)) / delta;
}

__device__ float do_fraction(float numer, float denom) {
	float result = 0.f; 

	if((numer == denom) && (numer != 0.f))
		result = 1.f;
	else if(denom != 0.f)
		result = numer / denom;

	return result;
}

// bins must be initialized to zero before calling get_bin_scores
__global__ void get_bin_scores(int nbins, int order,
	int nknots, float * knots, int nsamples, int nx, float * x, int pitch_x,
	float * bins, int pitch_bins)
{
	int
		col_x = blockDim.x * blockIdx.x + threadIdx.x;

	if(col_x >= nx)
		return;

	float
		* in_col = x + col_x * pitch_x,
		* bin_col = bins + col_x * pitch_bins;

	for(int k = 0; k < nsamples; k++, bin_col += nbins) {
		float z = in_col[k];
		int i0 = (int)floorf(z) + order - 1;
		if(i0 >= nbins)
			i0 = nbins - 1;

		bin_col[i0] = 1.f;
		for(int i = 2; i <= order; i++) {
			for(int j = i0 - i + 1; j <= i0; j++) {
				int
					dpom1 = j + i - 1, dpo = j + i, dp1 = j + 1;
				float
					ld = 0.f, rd = 0.f;
	
				ld = do_fraction(z - knots[j], knots[dpom1] - knots[j]);
				rd = do_fraction(knots[dpo] - z, knots[dpo] - knots[dp1]);

				float result;
				if (j + 1 >= nbins)
					result = ld * bin_col[j];
				else
					result = ld * bin_col[j] + rd * bin_col[j + 1];
				bin_col[j] = result;
			}
		}
	}
}

__global__ void get_entropy(int nbins, int nsamples, int nx,
	float * bin_scores, int pitch_bin_scores, float * entropies)
{
	int
		col_x = blockDim.x * blockIdx.x + threadIdx.x;

	if(col_x >= nx)
		return;

	float
		* in_col = bin_scores + col_x * pitch_bin_scores,
		entropy = 0.f, prob, logp;

    for(int i = 0; i < nbins; i++) {
        prob = 0.f;
        for(int j = 0; j < nsamples; j++)
            prob += in_col[j * nbins + i];
        prob /= (double) nsamples;

		if(prob <= 0.f)
			logp = 0.f;
		else
			logp = __log2f(prob);

        entropy += prob * logp;
    }
	entropies[col_x] = -entropy;
}

__global__ void get_joint_entropy(int nbins, int nsamples,
	int nx, float * x_bin_scores, int pitch_x_bin_scores,
	int ny, float * y_bin_scores, int pitch_y_bin_scores,
	float * joint_entropies, int pitch_joint_entropies)
{
	int
		col_x = blockDim.x * blockIdx.x + threadIdx.x,
		col_y = blockDim.y * blockIdx.y + threadIdx.y;

	if((col_x >= nx) || (col_y >= ny))
		return;

    float
		prob, logp, xy_entropy,
		* x_bins = x_bin_scores + col_x * pitch_x_bin_scores,
		* y_bins = y_bin_scores + col_y * pitch_y_bin_scores;
		
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
				logp = __log2f(prob);

            xy_entropy += prob * logp;
        }
    }
    xy_entropy = -xy_entropy;
	(joint_entropies + col_y * pitch_joint_entropies)[col_x] = xy_entropy;
}

void bSplineMutualInfo(int nbins, int order, int nsamples, int nx,
	const float * x, int ny, const float * y, float * out_mi)
{
	size_t
		pitch_x, pitch_y;
	int
		nknots, nblocks_x, nblocks_y;
	float
		* knots, 
		* mins_x, * maxes_x,
		* dmins_x, * dmaxes_x,
		* mins_y, * maxes_y,
		* dmins_y, * dmaxes_y,
		* dx, * dy,
		* dknots;

	mins_x = Calloc(nx, float);
	maxes_x = Calloc(nx, float);
	for(int i = 0; i < nx; i++) {
		const float * col = x + i * nsamples;
		mins_x[i] = get_min(nsamples, col);
		maxes_x[i] = get_max(nsamples, col);
	}

	mins_y = Calloc(ny, float);
	maxes_y = Calloc(ny, float);
	for(int i = 0; i < ny; i++) {
		const float * col = y + i * nsamples;
		mins_y[i] = get_min(nsamples, col);
		maxes_y[i] = get_max(nsamples, col);
	}

	cudaMalloc((void **)&dmins_x, nx * sizeof(float));
	cudaMemcpy(dmins_x, mins_x, nx * sizeof(float), cudaMemcpyHostToDevice);
	Free(mins_x);

	cudaMalloc((void **)&dmaxes_x, nx * sizeof(float));
	cudaMemcpy(dmaxes_x, maxes_x, nx * sizeof(float), cudaMemcpyHostToDevice);
	Free(maxes_x);

	cudaMalloc((void **)&dmins_y, ny * sizeof(float));
	cudaMemcpy(dmins_y, mins_y, ny * sizeof(float), cudaMemcpyHostToDevice);
	Free(mins_y);

	cudaMalloc((void **)&dmaxes_y, ny * sizeof(float));
	cudaMemcpy(dmaxes_y, maxes_y, ny * sizeof(float), cudaMemcpyHostToDevice);
	Free(maxes_y);

	checkCudaError("bSplineMutualInfoSingle: transfering extrema to the gpu");

	nknots = initKnots(nbins, order, &knots);
	float knot_max = knots[nknots - 1];

	cudaMallocPitch((void **)&dx, &pitch_x, nsamples * sizeof(float), nx);
	cudaMemcpy2D(dx, pitch_x, x, nsamples * sizeof(float),
		nsamples * sizeof(float), nx, cudaMemcpyHostToDevice);

	cudaMallocPitch((void **)&dy, &pitch_y, nsamples * sizeof(float), ny);
	cudaMemcpy2D(dy, pitch_y, y, nsamples * sizeof(float),
		nsamples * sizeof(float), ny, cudaMemcpyHostToDevice);

	checkCudaError("bSplineMutualInfoSingle: initializing vars on gpu");
	
	nblocks_x = nx / NTHREADS;
	if(nblocks_x * NTHREADS < nx)
		nblocks_x++;

	nblocks_y = ny / NTHREADS;
	if(nblocks_y * NTHREADS < ny)
		nblocks_y++;

	scale<<<nblocks_x, NTHREADS>>>(knot_max, nx, dmins_x, dmaxes_x, nsamples,
		dx, pitch_x / sizeof(float));

	scale<<<nblocks_y, NTHREADS>>>(knot_max, ny, dmins_y, dmaxes_y, nsamples,
		dy, pitch_y / sizeof(float));

	checkCudaError("bSplineMutualInfoSingle: scaling variables on gpu");

	cudaFree(dmins_x);
	cudaFree(dmaxes_x);
	cudaFree(dmins_y);
	cudaFree(dmaxes_y);

	float
		* dbins_x, * dbins_y;
	size_t
		pitch_bins_x, pitch_bins_y;

	cudaMallocPitch((void **)&dbins_x, &pitch_bins_x,
		nbins * nsamples * sizeof(float), nx);
	cudaMemset2D(dbins_x, pitch_bins_x, 0, nbins * nsamples * sizeof(float),
		nx);

	cudaMallocPitch((void **)&dbins_y, &pitch_bins_y,
		nbins * nsamples * sizeof(float), ny);
	cudaMemset2D(dbins_y, pitch_bins_y, 0, nbins * nsamples * sizeof(float),
		ny);

	cudaMalloc((void **)&dknots, nknots * sizeof(float));
	cudaMemcpy(dknots, knots, nknots * sizeof(float), cudaMemcpyHostToDevice);
	Free(knots);

	checkCudaError("bSplineMutualInfoSingle: initializing knots on gpu");

	get_bin_scores<<<nblocks_x, NTHREADS>>>(nbins, order, nknots, dknots,
		nsamples, nx, dx, pitch_x / sizeof(float),
		dbins_x, pitch_bins_x / sizeof(float));

	get_bin_scores<<<nblocks_y, NTHREADS>>>(nbins, order, nknots, dknots,
		nsamples, ny, dy, pitch_y / sizeof(float),
		dbins_y, pitch_bins_y / sizeof(float));

	checkCudaError("bSplineMutualInfoSingle: calculating bin scores on gpu");

	cudaFree(dknots);
	cudaFree(dx);
	cudaFree(dy);

	float
		* dentropies_x, * dentropies_y;

	cudaMalloc((void **)&dentropies_x, nx * sizeof(float));
	cudaMalloc((void **)&dentropies_y, ny * sizeof(float));

	checkCudaError("bSplineMutualInfoSingle: allocating gpu ram for entropies");

	get_entropy<<<nblocks_x, NTHREADS>>>(nbins, nsamples, nx, dbins_x,
		pitch_bins_x / sizeof(float), dentropies_x);

	get_entropy<<<nblocks_y, NTHREADS>>>(nbins, nsamples, ny, dbins_y,
		pitch_bins_y / sizeof(float), dentropies_y);

	checkCudaError("bSplineMutualInfoSingle: calculating entropy on gpu");

	size_t pitch_joint_entropies;
	float * djoint_entropies;
	cudaMallocPitch((void **)&djoint_entropies, &pitch_joint_entropies,
		ny * sizeof(float), nx);
	dim3
		gridDim(nblocks_x, nblocks_y), blockDim(NTHREADS, NTHREADS);
	get_joint_entropy<<<gridDim, blockDim>>>(nbins, nsamples,
		nx, dbins_x, pitch_bins_x / sizeof(float),
		ny, dbins_y, pitch_bins_y / sizeof(float),
		djoint_entropies, pitch_joint_entropies / sizeof(float));
	cudaFree(dbins_x);
	cudaFree(dbins_y);

	float
		* entropies_x = Calloc(nx, float), * entropies_y = Calloc(nx, float);

	cudaMemcpy2D(out_mi, ny * sizeof(float),
		djoint_entropies, pitch_joint_entropies, ny * sizeof(float), nx,
		cudaMemcpyDeviceToHost);
	cudaFree(djoint_entropies);

	cudaMemcpy(entropies_x, dentropies_x, nx * sizeof(float),
		cudaMemcpyDeviceToHost);
	cudaFree(dentropies_x);

	cudaMemcpy(entropies_y, dentropies_y, ny * sizeof(float),
		cudaMemcpyDeviceToHost);
	cudaFree(dentropies_y);

	checkCudaError("bSplineMutualInfoSingle: retrieving results from gpu");

	for(int j = 0; j < ny; j++) {
		float * out_col = out_mi + j * nx;
		for(int i = 0; i < nx; i++)
			out_col[i] = entropies_x[i] + entropies_y[j] - out_col[i];
	}
}

void rBSplineMutualInfo(int * nBins, int * splineOrder, int * nsamples,
	int * rowsA, const float * A, int * rowsB, const float * B, 
	float * mutualInfo)
{
	bSplineMutualInfo(*nBins, *splineOrder, *nsamples, *rowsA, A, *rowsB, B,
		mutualInfo);
}
