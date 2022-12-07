#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace GPU
{
	__host__ float* gpuKMeans(float* points, int N, int n, int K, int max_iterations);

	// common

	__device__ float squareDistance(float* p, float* q, int n);

	__global__ void divideCentroidCoordinates(float* centroids, int* cluster_sizes, int n, int K);

	__global__ void assignPointsToClusters(float* points, float* centroids, int* assignments, int N, int n, int K);

	// naive implementation

	__global__ void updateCentroidsNaive(float* points, float* centroids, int* assignments, int* cluster_sizes, int N, int n, int K);

	// better implementation

	__global__ void prepareData(float* points, int* assignments, float* centroids_data, int* cluster_sizes_data, int N, int n, int K);

	__global__ void sumData(float* centroids_data, int* cluster_sizes_data, int data_size, int n, int K);

}