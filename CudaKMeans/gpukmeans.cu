#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cooperative_groups.h>

#include "gpukmeans.cuh"
#include "utils.h"

namespace GPU
{
    __host__ float* gpuKMeans(float* points, int N, int n, int K, int max_iterations)
    {
        float* centroids = sampleCentroids(N, n, K);

        // init
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            return NULL;
        }

        // prepare points to be used by GPU
        float* d_points = 0;
        cudaStatus = cudaMalloc((void**)&d_points, N * n * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return NULL;
        }
        cudaStatus = cudaMemcpy(d_points, points, N * n * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            cudaFree(d_points);
            return NULL;
        }

        // prepare centroids to be used by GPU
        float* d_centroids = 0;
        cudaStatus = cudaMalloc((void**)&d_centroids, K * n * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return NULL;
        }
        cudaStatus = cudaMemcpy(d_centroids, centroids, K * n * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            cudaFree(d_centroids);
            return NULL;
        }

        // prepare assignments
        int* assignments = new int[N];
        int* d_assignments = 0;
        cudaStatus = cudaMalloc((void**)&d_assignments, N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate memory for assignments");
            return NULL;
        }

        // prepare cluster sizes
        int* cluster_sizes = new int[K];
        int* d_cluster_sizes = 0;
        cudaStatus = cudaMalloc((void**)&d_cluster_sizes, K * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate memory for cluster sizes");
            return NULL;
        }

        // calculate kernel parameters
        dim3 gridDim((unsigned int)ceilf((float)N / THREADS_PER_BLOCK));
        dim3 blockDim(THREADS_PER_BLOCK);

        // main algorithm
        for (int iteration = 0; iteration < max_iterations; iteration++)
        {
            // calculate assignments
            assignPointsToClusters<<<gridDim, blockDim>>>(d_points, d_centroids, d_assignments, N, n, K);

            // check if enough points changed clusters
            /*cudaStatus = cudaMemcpy(assignments, d_assignments, N * sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }*/


            // DEBUG print centroid coordinates
            /*cudaStatus = cudaMemcpy(centroids, d_centroids, K * n * sizeof(float), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }
            std::cout << "Centroids after " << iteration << " iterations" << std::endl;
            for (int i = 0; i < K; i++)
            {
                std::cout << centroids[i * n] << " " << centroids[i * n + 1] << std::endl;
            }*/

            // update clusters

            cudaMemset(d_centroids, 0.0, K * n * sizeof(float));
            cudaMemset(d_cluster_sizes, 0, K * sizeof(int));

            // reserve memory in shared for assignments and points
            // THREADS_PER_BLOCK is the number of points processed by a block
            // so we need THREADS_PER_BLOCK assignements (int)
            // and THREADS_PER_BLOCK * n coordinates (floats)
            updateCentroidsNaive<<<gridDim, blockDim, THREADS_PER_BLOCK * sizeof(int) + THREADS_PER_BLOCK * n * sizeof(float)>>>
                (d_points, d_centroids, d_assignments, d_cluster_sizes, N, n, K);

            
            // Divide coordinates by cluster sizes
            // TODO Parallelise this using CUDA

            divideCentroidCoordinates<<<1, K>>>(d_centroids, d_cluster_sizes, n, K);
            /*cudaStatus = cudaMemcpy(centroids, d_centroids, K * n * sizeof(float), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }
            cudaStatus = cudaMemcpy(cluster_sizes, d_cluster_sizes, K * sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }

            for (int i = 0; i < K; i++)
            {
                for (int dimension = 0; dimension < n; dimension++)
                {
                    centroids[i * n + dimension] = centroids[i * n + dimension] / cluster_sizes[i];
                }
            }

            cudaStatus = cudaMemcpy(d_centroids, centroids, K * n * sizeof(float), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }*/



            // DEBUG print cluster sizes
            /*cudaStatus = cudaMemcpy(cluster_sizes, d_cluster_sizes, K * sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }
            std::cout << "Cluster sizes after " << iteration + 1 << " iterations" << std::endl;
            for (int i = 0; i < K; i++)
            {
                std::cout << cluster_sizes[i] << std::endl;
            }*/
        }
        cudaDeviceSynchronize();

        cudaStatus = cudaMemcpy(centroids, d_centroids, K * n * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
        }
        
        // clean-up

        delete[] assignments;
        delete[] cluster_sizes;

        cudaFree(d_points);
        cudaFree(d_centroids);
        cudaFree(d_assignments);
        cudaFree(d_cluster_sizes);

        cudaDeviceReset();

        return centroids;
    }

    __device__ float squareDistance(float* p, float* q, int n)
    {
        float distance = 0;
        for (int i = 0; i < n; i++)
        {
            distance += (p[i] - q[i]) * (p[i] - q[i]);
        }
        return distance;
    }

    __global__ void assignPointsToClusters(float* points, float* centroids, int* assignments, int N, int n, int K)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int point_idx = idx * n;

        if (idx >= N)
        {
            return;
        }

        float min_distance = MAX_FLOAT;
        int closest_centroid = 0;

        for (int centroid = 0; centroid < K; centroid++)
        {
            float distance = squareDistance(&points[point_idx], &centroids[centroid * n], n);

            if (distance < min_distance)
            {
                min_distance = distance;
                closest_centroid = centroid;
            }
        }

        assignments[idx] = closest_centroid;
    }

    __global__ void updateCentroidsNaive(float* points, float* centroids, int* assignments, int* cluster_sizes, int N, int n, int K)
    {
        const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int local_idx = threadIdx.x;

        const int global_point_idx = global_idx * n;
        const int local_point_idx = local_idx * n;         

        if (global_idx >= N)
        {
            return;
        }

        extern __shared__ float s_points[];
        extern __shared__ int s_assignments[];
        int memory_offset = THREADS_PER_BLOCK; // used to offset pointers when using s_points

        // copy corresponding point to shared memory
        for (int dimension = 0; dimension < n; dimension++)
        {
            s_points[memory_offset + local_point_idx + dimension] = points[global_point_idx + dimension];
        }

        // copy assignments to shared memory
        s_assignments[local_idx] = assignments[global_idx];

        __syncthreads();

        if (local_idx == 0)
        {
            // initialise arrays for summing up point coordinates and counting cluster sizes
            float* b_coordinates_sums = new float[K * n];
            for (int i = 0; i < K * n; i++)
            {
                b_coordinates_sums[i] = 0.0;
            }

            int* b_cluster_sizes = new int[K];
            for (int i = 0; i < K; i++)
            {
                b_cluster_sizes[i] = 0;
            }

            // sum coordinates of points belonging to the block
            for (int point = 0; point < blockDim.x; point++)
            {
                if (global_idx + point >= N)
                {
                    break;
                }
                int centroid = s_assignments[point];
                b_cluster_sizes[centroid]++;

                // add point coordinates to the relevant sum
                for (int dimension = 0; dimension < n; dimension++)
                {
                    b_coordinates_sums[centroid * n + dimension] += s_points[memory_offset + point * n + dimension];
                }
            }

            // sum the intermediate sums globally
            for (int centroid = 0; centroid < K; centroid++)
            {
                atomicAdd(&cluster_sizes[centroid], b_cluster_sizes[centroid]);
                for (int dimension = 0; dimension < n; dimension++)
                {
                    atomicAdd(&centroids[centroid * n + dimension], b_coordinates_sums[centroid * n + dimension]);
                }
            }

            delete[] b_coordinates_sums;
            delete[] b_cluster_sizes;
        }

        __syncthreads();
    }



    __global__ void divideCentroidCoordinates(float* centroids, int* cluster_sizes, int n, int K)
    {
        int centroid = threadIdx.x;
        for (int dimension = 0; dimension < n; dimension++)
        {
            centroids[centroid * n + dimension] = centroids[centroid * n + dimension] / cluster_sizes[centroid];
        }
    }
}