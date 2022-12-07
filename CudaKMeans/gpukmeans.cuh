#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PR_MODE int

namespace GPU
{
	const PR_MODE PROPER_PARALLEL_REDUCTION = 0;
	const PR_MODE SIMPLIFIED_PARALLEL_REDUCTION = 1;

	__host__ float* gpuKMeans(float* points, int N, int n, int K, int max_iterations, PR_MODE parallel_reduction_mode = PROPER_PARALLEL_REDUCTION);

	// common

	__global__ void divideCentroidCoordinates(float* centroids, int* cluster_sizes, int n, int K);

	__global__ void assignPointsToClusters(float* points, float* centroids, int* assignments, int N, int n, int K);

	// naive implementation

	__global__ void updateCentroidsNaive(float* points, float* centroids, int* assignments, int* cluster_sizes, int N, int n, int K);

	// better implementation

	__device__ void warpReduce(
		volatile float* s_centroids_data, volatile int* s_cluster_sizes_data,
		const int local_idx, const int global_idx,
		const int data_size, const int K, const int term_size, const int memory_offset);

	__global__ void prepareData(float* points, int* assignments, float* centroids_data, int* cluster_sizes_data, int N, int n, int K);

	__global__ void sumData(float* centroids_data, int* cluster_sizes_data, int data_size, int n, int K);

}