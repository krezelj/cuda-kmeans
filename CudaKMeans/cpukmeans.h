#pragma once

namespace CPU
{
	float* cpuKMeans(float* points, int N, int n, int K, int max_iterations = 100);

	void assignPointsToClusters(float* points, float* centroids, int* assignments, int N, int n, int K);

	void updateCentroids(float* points, float* centroids, int* assignments, int N, int n, int K);

	float squareDistance(float* p, float* q, int n);
}

