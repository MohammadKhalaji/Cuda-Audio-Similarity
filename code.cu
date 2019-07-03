
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>
#include <omp.h>
#include <dirent.h>
#include <math.h>
#include <string.h>


// the type of data read from files
typedef struct {
	int *elements;
	int size;
} Array;


// GLOBALS
char* musicPath; // a string containing the path in which musics exist
char* samplePath; // a string containing the path in which samoles exist
int numberOfMusics;
int numberOfSamples;
char **musicPaths;
char **samplePaths;

// Globals for reduction
#define blockSize 1024
long double *d_array[1000];
int level = 0;


// this function reads a single file into Array *a
void read_file(char *path, Array *a);

// this function is used for testing
void test_init(Array *a);

// this function computes the fourier transform of Array *a
cufftComplex *fft(Array *a);

// this function returns number of files in a certain path
int number_of_files(char *path);

// this function finds all file paths in char *path and puts them in char **res
void init_lists(char *path, char **res);

// char **path contains all file paths to be read
// this function reads all files with paths given in char **path
void read_files(char **paths, Array **dest, int size);

// this function returns power spectrum of Array *a
// power spectrum = sqrt(real^2 + imag^2)
long double *real_fft(Array *a);

// this kernel is used for computing the power spectrum
// out[i] = sqrt(in[i].real^2 + in[i].imag^2)
__global__ void real_kernel(cufftComplex *in, long double *out, int size);

// this function is used after calling each cuda function
void check_errors(cudaError_t status, char *line);

// serial comparison
long double compare(Array *sample, Array *music);

// serial cosine similarity
long double similarity(long double *sampleFft, long double *musicSliceFft, int size);

// this kernel does the element-by-element multiplication
__global__ void mul_kernel(long double *in1, long double *in2, long double *out, int size);

// parallel comparison
long double compare_parallel(Array *sample, Array *music);

// parallel cosine similarity
long double cosine_similarity(long double *sampleFft, long double *musicSliceFft, int size);

// this is the helper function to do reduction on a certain array
long double reduce(long double *arr, int size);

// this kernel does the reduction :)
__global__ void reduction_kernel(long double *g_idata, long double *g_odata, int size);

// serial reduction code
long double serial_reduce(long double *arr, int size);


int main(int argc, char* argv[])
{

	if (argc != 3) {
		printf("Invalid number of arguments\n");
		return 1;
	}

	// read the arguments
	musicPath = argv[1];
	samplePath = argv[2];



	// fill lists with paths
	omp_set_nested(1);
	#pragma omp parallel num_threads(2)
	{
		int id = omp_get_thread_num();

		if (id == 0) {
			numberOfMusics = number_of_files(musicPath);
			musicPaths = (char **)malloc(sizeof(char *) * numberOfMusics);
			init_lists(musicPath, musicPaths);
		}
		else {
			numberOfSamples = number_of_files(samplePath);
			samplePaths = (char **)malloc(sizeof(char *) * numberOfSamples);
			init_lists(samplePath, samplePaths);

		}
	}

	// alocate placeholders for musics and samples
	Array **musics = (Array **)malloc(numberOfMusics * sizeof(Array *));
	Array **samples = (Array **)malloc(numberOfSamples * sizeof(Array *));


	// allocation...
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < numberOfMusics; i++) {
		musics[i] = (Array *)malloc(sizeof(*musics[i]));
	}


	// allocation...
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < numberOfSamples; i++) {
		samples[i] = (Array *)malloc(sizeof(*samples[i]));
	}

	#pragma omp parallel num_threads(2)
	{
		int id = omp_get_thread_num();
		if (id == 0) {
			read_files(musicPaths, musics, numberOfMusics);
		}
		else {
			read_files(samplePaths, samples, numberOfSamples);
		}
	}

	Array *music;
	Array *sample;
	long double current, best = -INFINITY;
	int bestIndex = 0;

	printf("\n------------------------ Musics Read -------------------------\n");
	for (int i = 0; i < numberOfMusics; i++) {
		printf("%s\n", musicPaths[i]);
	}

	printf("\n----------------------- Samples Read ------------------------\n");
	for (int i = 0; i < numberOfSamples; i++) {
		printf("%s\n", samplePaths[i]);
	}

	printf("\n-------------------------------------------------------------\n");


	for (int j = 0; j < numberOfSamples; j++) {
		sample = samples[j];
		best = -INFINITY;
		for (int i = 0; i < numberOfMusics; i++) {
			music = musics[i];
			current = compare_parallel(sample, music);
			if (current > best) {
				best = current;
				bestIndex = i;
			}
		}
		if (best > 0.7)
			printf("%s >>> %s\n", samplePaths[j], musicPaths[bestIndex]);
		else
			printf("%s >>> Not Found\n", samplePaths[j]);
		printf("\n-------------------------------------------------------------\n");

	}

	/*
	Array *music = (Array *)malloc(sizeof(*music));
	Array *sample = (Array *)malloc(sizeof(*sample));
	long double current, best = -INFINITY;
	int bestIndex = 0;

	for (int i = 0; i < numberOfMusics; i++) {
		read_file(musicPaths[i], music);
		//musicFft = real_fft(music);

		for (int j = 0; j < numberOfSamples; j++) {
			read_file(samplePaths[j], sample);
			//sampleFft = real_fft(sample);

			// COMPARE
			//compare(sample, music);
			current = compare_parallel(sample, music);

			if (current > best) {
				best = current;
				bestIndex = j;
			}
			//printf("%s, %s: %f\n", musicPaths[i], samplePaths[j], current);
			//free(sampleFft);
			free(sample->elements);
		}

		if (best > 0.7)
			printf("%s >>> %s\n", musicPaths[i], samplePaths[bestIndex]);
		else
			printf("%s >>> Not Found\n", musicPaths[i]);

		//free(musicFft);
		free(music->elements);
	}
	*/

	return 0;
}

void read_file(char *path, Array *a) {
	printf("\n");
	printf("started reading %s\n", path);
	FILE* fp = fopen(path, "r");
	int count = 0;
	int i = 0;
	int num;

	while (fscanf(fp, " %d", &num) == 1) {
		count++;
	}
	fclose(fp);
	fp = fopen(path, "r");
	a->elements = (int*)malloc(count * sizeof(int));
	a->size = count;

	while (fscanf(fp, " %d", &num) == 1) {
		a->elements[i] = num;
		i++;
	}
	printf("finished reading %s\n", path);

	fclose(fp);
}

void test_init(Array *a) {
	a->size = 100;
	a->elements = (int *)malloc(a->size * sizeof(int));
	for (int i = 0; i < a->size; i++) {
		a->elements[i] = i;
	}
}

cufftComplex *fft(Array *a) {

	// Select the GPU
	cudaSetDevice(0);

	cufftHandle plan;
	cufftComplex *d_data, *h_data;

	// h_data is used to convert int* in Array object to a cufftComplex type
	h_data = (cufftComplex *)malloc(a->size * sizeof(cufftComplex));

	// d_data is the copy of h_data in the GPU
	cudaMalloc((void **)&d_data, a->size * sizeof(cufftComplex));


	// Convert Array object to cufftComplex type
	for (int i = 0; i < a->size; i++) {
		h_data[i].x = a->elements[i];
		h_data[i].y = 0;
	}

	// copy h_data to GPU
	cudaMemcpy(d_data, h_data, a->size * sizeof(cufftComplex), cudaMemcpyHostToDevice);

	// Compute Fourier Transform in GPU
	cufftPlan1d(&plan, a->size, CUFFT_C2C, 1);
	cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

	// Wait for GPU operation to completes
	cudaDeviceSynchronize();

	// Now copy the result back to host
	cudaMemcpy(h_data, d_data, a->size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	cufftDestroy(plan);
	cudaFree(d_data);
	return h_data;
}

long double *real_fft(Array *a) {
	cudaError_t status;
	cufftComplex *fourier = fft(a);

	/*for (int i = 0; i < a->size; a++) {
		printf("%f %f\n", fourier[i].x, fourier[i].y);
	}
	printf("\n");
*/


	long double *out = (long double *)malloc(sizeof(long double) * a->size);
	long double *d_out;
	cufftComplex *d_fourier;

	status = cudaMalloc((void **)&d_out, a->size * sizeof(long double));
	check_errors(status, "cudaMalloc(d_out)");

	status = cudaMalloc((void **)&d_fourier, a->size * sizeof(cufftComplex));
	check_errors(status, "cudaMalloc(d_fourier)");

	status = cudaMemcpy(d_fourier, fourier, a->size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	check_errors(status, "cudaMemcpy(d_fourier, fourier)");

	int n_blocks = ceil((long double)a->size / 1024.0);
	int n_threads = (n_blocks > 1) ? 1024 : a->size;

	real_kernel << <n_blocks, n_threads >> > (d_fourier, d_out, a->size);
	check_errors(cudaGetLastError(), "kernel real_fft");

	status = cudaDeviceSynchronize();
	check_errors(status, "cudaDeviceSync");

	status = cudaMemcpy(out, d_out, a->size * sizeof(long double), cudaMemcpyDeviceToHost);
	check_errors(status, "cudaMemcpy(out, d_out)");

	status = cudaFree(d_out);
	check_errors(status, "cudaFree(d_out)");

	status = cudaFree(d_fourier);
	check_errors(status, "cudaFree(d_fourier)");

	free(fourier);
	return out;
}

__global__ void real_kernel(cufftComplex *in, long double *out, int size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		out[i] = sqrt(in[i].x * in[i].x + in[i].y * in[i].y);
	}
}


void check_errors(cudaError_t status, char *line) {
	if (status != cudaSuccess)
		printf("\n%s - %s", cudaGetErrorString(status), line);
}


long double compare(Array *sample, Array *music) {
	int iters = ceil((long double)music->size / (long double)sample->size);
	Array *slice = (Array *)malloc(sizeof(*slice));
	long double *sampleFft = real_fft(sample);
	long double *musicFft;
	int sliceSize = sample->size;
	int residual = iters * sliceSize - music->size;
	int start;
	int residualStart;
	int flag = 0;
	slice->size = sliceSize;

	for (int k = 0; k < iters; k++) {
		start = k * sliceSize;
		slice->elements = &music->elements[start];
		if (k == iters - 1) {
			slice->elements = (int *)malloc(sliceSize * sizeof(int));
			flag = 1;
			residualStart = sliceSize - residual;
			for (int l = 0; l < sliceSize; l++)
				slice->elements[l] = (l >= residualStart) ? 0 : music->elements[l + (iters - 1)*sliceSize];
		}
		musicFft = real_fft(slice);
		similarity(sampleFft, musicFft, sliceSize);

		flag ? free(slice) : 0;
		free(musicFft);
	}

	free(sampleFft);
	return 0.0;
}


long double compare_parallel(Array *sample, Array *music) {
	int iters = ceil((long double)music->size / (long double)sample->size);
	Array *slice = (Array *)malloc(sizeof(*slice));
	long double *sampleFft = real_fft(sample);
	long double *musicFft;
	int sliceSize = sample->size;
	int residual = iters * sliceSize - music->size;
	int start;
	int residualStart;
	int flag = 0;
	long double sim = 0.0;
	long double max = -INFINITY;
	slice->size = sliceSize;

	for (int k = 0; k < iters; k++) {
		flag = 0;
		start = k * sliceSize;
		slice->elements = &music->elements[start];
		if (k == iters - 1) {
			slice->elements = (int *)malloc(sliceSize * sizeof(int));
			flag = 1;
			residualStart = sliceSize - residual;
			#pragma omp parallel for num_threads(4)
			for (int l = 0; l < sliceSize; l++) {
				slice->elements[l] = (l >= residualStart) ? 0 : music->elements[l + (iters - 1)*sliceSize];
			}
		}
		musicFft = real_fft(slice);

		sim = cosine_similarity(sampleFft, musicFft, sliceSize);
		if (sim > max) {
			max = sim;
		}

		flag ? free(slice) : 0;
		free(musicFft);
	}

	free(sampleFft);
	return max;
}

long double cosine_similarity(long double *sampleFft, long double *musicSliceFft, int size) {

	long double *d_in1, *d_in2, *d_out;
	long double *dot, *norm1, *norm2;

	dot = (long double *)malloc(size * sizeof(long double));
	norm1 = (long double *)malloc(size * sizeof(long double));
	norm2 = (long double *)malloc(size * sizeof(long double));

	cudaError_t status;
	cudaSetDevice(0);
	status = cudaMalloc((void **)&d_in1, size * sizeof(long double));
	check_errors(status, "cudaMalloc(in1)");
	status = cudaMalloc((void **)&d_in2, size * sizeof(long double));
	check_errors(status, "cudaMalloc(in2)");
	status = cudaMalloc((void **)&d_out, size * sizeof(long double));
	check_errors(status, "cudaMalloc(out)");
	status = cudaMemcpy(d_in1, sampleFft, size * sizeof(long double), cudaMemcpyHostToDevice);
	check_errors(status, "cudaMemcpy(in1)");
	status = cudaMemcpy(d_in2, musicSliceFft, size * sizeof(long double), cudaMemcpyHostToDevice);
	check_errors(status, "cudaMemcpy(in2)");

	int n_blocks = ceil((long double)size / 1024.0);
	int n_threads = (n_blocks > 1) ? 1024 : size;

	mul_kernel << <n_blocks, n_threads >> > (d_in1, d_in2, d_out, size);
	check_errors(cudaGetLastError(), "kernel(in1, in2)");

	status = cudaDeviceSynchronize();
	check_errors(status, "cudaDeviceSync");

	status = cudaMemcpy(dot, d_out, size * sizeof(long double), cudaMemcpyDeviceToHost);
	check_errors(status, "cudaMemcpy(dot, d_out)");

	mul_kernel << <n_blocks, n_threads >> > (d_in1, d_in1, d_out, size);
	check_errors(cudaGetLastError(), "kernel(in1, in1)");


	status = cudaDeviceSynchronize();
	check_errors(status, "cudaDeviceSync");

	status = cudaMemcpy(norm1, d_out, size * sizeof(long double), cudaMemcpyDeviceToHost);
	check_errors(status, "cudaMemcpy(norm1, d_out)");

	mul_kernel << <n_blocks, n_threads >> > (d_in2, d_in2, d_out, size);
	check_errors(cudaGetLastError(), "kernel(in2, in2)");

	status = cudaDeviceSynchronize();
	check_errors(status, "cudaDeviceSync");

	status = cudaMemcpy(norm2, d_out, size * sizeof(long double), cudaMemcpyDeviceToHost);
	check_errors(status, "cudaMemcpy(norm2, d_out)");



	long double f_dot = reduce(dot, size);
	long double f_norm1 = sqrt(reduce(norm1, size));
	long double f_norm2 = sqrt(reduce(norm2, size));
	

	status = cudaFree(d_in1);
	check_errors(status, "cudaFree(in1)");
	status = cudaFree(d_in2);
	check_errors(status, "cudaFree(in2)");
	status = cudaFree(d_out);
	check_errors(status, "cudaFree(out)");

	
	long double res = (f_dot) / (f_norm1 * f_norm2);
	

	return res;
}



__global__ void mul_kernel(long double *in1, long double *in2, long double *out, int size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
		out[i] = in1[i] * in2[i];
}

long double reduce(long double *arr, int size) {
	int newsize = pow(2, ceil(log(size) / log(2)));
	arr = (long double *)realloc(arr, newsize * sizeof(long double));

	#pragma omp parallel for num_threads(6)
	for (int i = size; i < newsize; i++)
		arr[i] = 0;

	cudaError_t status;
	status = cudaSetDevice(0);
	check_errors(status, "Dev Set");

	int i = newsize;
	int dcount = 0;
	while (i != 0) {
		status = cudaMalloc((void **)&d_array[level], i * sizeof(long double));
		//printf("allocated level %d with size %d\n", level, i);
		dcount++;
		check_errors(status, "cudaMalloc(d_array[level])");

		if (i == 1) {
			i = 0;
		}
		else {
			i = ((i - 1) / blockSize) + 1;
		}

		level++;
	}

	status = cudaMemcpy(d_array[0], arr, newsize * sizeof(long double), cudaMemcpyHostToDevice);
	check_errors(status, "Memcpy(d_array[0], arr)");

	int current = newsize;
	int next = ((current - 1) / blockSize) + 1;
	int counter = 0;

	while (current != 1) {
		reduction_kernel << <next, blockSize/2 >> > (d_array[counter], d_array[counter + 1], current);
		//printf("called kernel for level %d and %d\n", counter, counter+1);
		check_errors(cudaGetLastError(), "kernel");
		current = next;
		next = ((current - 1) / blockSize) + 1;
		counter++;
	}

	status = cudaDeviceSynchronize();
	check_errors(status, "Dev Sync");
	cudaMemcpy(arr, d_array[level - 1], sizeof(long double), cudaMemcpyDeviceToHost);
	
	for (int j = 0; i < dcount; i++) {
		status = cudaFree(d_array[i]);
		check_errors(status, "cudaFree");
	}
	level = 0;
	float res = arr[0];
	free(arr);
	return res;
}

long double serial_reduce(long double *arr, int size) {
	long double res = 0.0;
	for (int i = 0; i < size; i++)
		res += arr[i];
	return res;
}

__global__ void reduction_kernel(long double *g_idata, long double *g_odata, int size)
{
	__shared__ long double sdata[blockSize];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	if (i >= size)
		sdata[tid] = 0;
	else if (i + blockDim.x >= size)
		sdata[tid] = g_idata[i];
	else
		sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
	{
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32];
		__syncwarp();
		sdata[tid] += sdata[tid + 16];
		__syncwarp();
		sdata[tid] += sdata[tid + 8];
		__syncwarp();
		sdata[tid] += sdata[tid + 4];
		__syncwarp();
		sdata[tid] += sdata[tid + 2];
		__syncwarp();
		sdata[tid] += sdata[tid + 1];
		__syncwarp();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


long double similarity(long double *sampleFft, long double *musicSliceFft, int size) {

	long double res = 0.0;
	long double dot = 0.0;
	long double norm1 = 0.0;
	long double norm2 = 0.0;

	for (int i = 0; i < size; i++) {
		dot += sampleFft[i] * musicSliceFft[i];
		norm1 += sampleFft[i] * sampleFft[i];
		norm2 += musicSliceFft[i] * musicSliceFft[i];
	}

	norm1 = sqrt(norm1);
	norm2 = sqrt(norm2);
	res = dot / (norm1 * norm2);

	return res;
}


int number_of_files(char *path) {
	DIR *dir;
	struct dirent *ent;
	char *dot;
	int count = 0;
	if ((dir = opendir(path)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			dot = strrchr(ent->d_name, '.');
			if (dot && !strcmp(dot, ".txt")) {
				count++;
			}
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
	}

	return count;
}

void init_lists(char *path, char **res) {
	int i = 0;
	char currentPath[500];
	DIR *dir;
	struct dirent *ent;
	char *dot;
	if ((dir = opendir(path)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			dot = strrchr(ent->d_name, '.');
			if (dot && !strcmp(dot, ".txt")) {
				strcpy(currentPath, path);
				strcat(currentPath, "\\");
				strcat(currentPath, ent->d_name);
				res[i] = (char *)malloc(strlen(currentPath) * sizeof(char));
				strcpy(res[i], currentPath);
				i++;
			}
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");

	}

}

void read_files(char **paths, Array **dest, int size) {
	char *currentPath;
#pragma omp parallel for num_threads(4) private(currentPath)
	for (int i = 0; i < size; i++) {
		currentPath = paths[i];
		read_file(currentPath, dest[i]);
	}

}


