#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace GPU
{
	__host__ float* gpuKMeans(float* points, int N, int n, int K, int max_iterations);

	__global__ void assignPointsToClusters(float* points, float* centroids, int* assignments, int N, int n, int K);

	__global__ void updateCentroidsNaive(float* points, float* centroids, int* assignments, int* cluster_sizes, int N, int n, int K);

	__device__ float squareDistance(float* p, float* q, int n);

	__global__ void divideCentroidCoordinates(float* centroids, int* cluster_sizes, int n, int K);
}