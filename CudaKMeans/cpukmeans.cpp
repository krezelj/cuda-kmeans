#include "cpukmeans.h"
#include "utils.h"
#include <iostream>


namespace CPU
{
    float* cpuKMeans(float* points, int N, int n, int K, int max_iterations)
    {
        float* centroids = sampleCentroids(N, n, K);
        int* assignments = new int[N];

        for (int iteration = 0; iteration < max_iterations; iteration++)
        {
            assignPointsToClusters(points, centroids, assignments, N, n, K);
            updateCentroids(points, centroids, assignments, N, n, K);
        }

        return centroids;
    }

    float squareDistance(float* p, float* q, int n)
    {
        float distance = 0;
        for (int dimension = 0; dimension < n; dimension++)
        {
            distance += (p[dimension] - q[dimension]) * (p[dimension] - q[dimension]);
        }
        return distance;
    }


    void assignPointsToClusters(float* points, float* centroids, int* assignments, int N, int n, int K)
    {
        for (int point = 0; point < N; point++)
        {
            float min_distance = MAX_FLOAT;
            for (int centroid = 0; centroid < K; centroid++)
            {
                float distance = squareDistance(&points[point * n], &centroids[centroid * n], n);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    assignments[point] = centroid;
                }
            }
        }
    }


    void updateCentroids(float* points, float* centroids, int* assignments, int N, int n, int K)
    {
        // init
        float* new_centroids = new float[K * n];
        int* cluster_sizes = new int[K];

        for (int i = 0; i < K; i++)
        {
            cluster_sizes[i] = 0;
        }
        for (int i = 0; i < K * n; i++)
        {
            new_centroids[i] = 0;
        }

        // sum up coordinates
        for (int point = 0; point < N; point++)
        {
            int centroid = assignments[point];
            cluster_sizes[centroid]++;
            for (int dimension = 0; dimension < n; dimension++)
            {
                new_centroids[centroid * n + dimension] += points[point * n + dimension];
            }
        }

        // divide and copy
        for (int centroid = 0; centroid < K; centroid++)
        {
            for (int dimension = 0; dimension < n; dimension++)
            {
                centroids[centroid * n + dimension] = new_centroids[centroid * n + dimension] / cluster_sizes[centroid];
            }
        }

        delete[] new_centroids;
        delete[] cluster_sizes;
    }

}