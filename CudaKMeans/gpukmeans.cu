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

        // calculate kernel parameters
        dim3 gridDim((unsigned int)ceilf((float)N / THREADS_PER_BLOCK));
        dim3 blockDim(THREADS_PER_BLOCK);

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

        // prepare data array for summing cluster coordinates
        // float* points_data = new float[K * n * N];
        float* d_centroids_data = 0;
        cudaStatus = cudaMalloc((void**)&d_centroids_data, K * n * N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate memory for points data");
            return NULL;
        }

        // prepare data array for summing cluster sizes
        int* d_cluster_sizes_data = 0;
        cudaStatus = cudaMalloc((void**)&d_cluster_sizes_data, K * N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate memory for assignments data");
            return NULL;
        }

        // main algorithm
        for (int iteration = 0; iteration < max_iterations; iteration++)
        {
            // calculate assignments
            assignPointsToClusters<<<gridDim, blockDim, K * n * sizeof(float)>>>(d_points, d_centroids, d_assignments, N, n, K);

            // check if enough points changed clusters
            /*cudaStatus = cudaMemcpy(assignments, d_assignments, N * sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }*/

            // update clusters


            // ------------------------------------- naive version --------------------------------------------
            // // reset centroids and cluster sizes to prepare them for summation
            //cudaMemset(d_centroids, 0.0, K * n * sizeof(float));
            //cudaMemset(d_cluster_sizes, 0, K * sizeof(int));

            //// reserve shared memory for assignments and points, we need:
            //// THREADS_PER_BLOCK        assignements (int) +
            //// THREADS_PER_BLOCK * n    coordinates (floats)
            //updateCentroidsNaive<<<gridDim, blockDim, THREADS_PER_BLOCK * sizeof(int) + THREADS_PER_BLOCK * n * sizeof(float)>>>
            //    (d_points, d_centroids, d_assignments, d_cluster_sizes, N, n, K);

            // -------------------------------------------------------------------------------------------------

            prepareData<<<gridDim, blockDim>>>(d_points, d_assignments, d_centroids_data, d_cluster_sizes_data, N, n, K);

            int data_size = N;
            while (data_size > 1)
            {
                unsigned int blocks = ceilf((float)data_size / THREADS_PER_BLOCK);
                sumData<<<blocks, blockDim, K * n * THREADS_PER_BLOCK * sizeof(float) + K * THREADS_PER_BLOCK * sizeof(int)>>>
                    (d_centroids_data, d_cluster_sizes_data, data_size, n, K);
                data_size = blocks;
            }

            // move data
            cudaStatus = cudaMemcpy(d_centroids, d_centroids_data, K * n * sizeof(float), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }

            cudaStatus = cudaMemcpy(d_cluster_sizes, d_cluster_sizes_data, K * sizeof(int), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }

            divideCentroidCoordinates<<<1, K>>>(d_centroids, d_cluster_sizes, n, K);

            /*float* points_data = new float[K * n];
            cudaStatus = cudaMemcpy(points_data, d_centroids_data, K * n * sizeof(float), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }
            for (int i = 0; i < K * n; i++)
            {
                std::cout << points_data[i] << std::endl;
            }*/

            /*int* ass = new int[K];
            cudaStatus = cudaMemcpy(ass, d_assignments_data, K * sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
            }
            for (int i = 0; i < K; i++)
            {
                std::cout << ass[i] << std::endl;
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
        const int local_idx = threadIdx.x;
        const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int point_idx = global_idx * n;

        if (global_idx >= N)
        {
            return;
        }

        // copy centroid data to shared memory
        // we assume that N >= K (typically N >> K so it's a reasonable assumption)
        // we also must assume that K <= THREADS_PER_BLOCK
        // if K > TPB we should copy more than one centroid per thread
        // but we will assume K < TPB
        extern __shared__ float s_centroids[]; // size is K * n
        if (local_idx < K)
        {
            for (int dimension = 0; dimension < n; dimension++)
            {
                s_centroids[local_idx * n + dimension] = centroids[local_idx * n + dimension];
            }
        }


        float min_distance = MAX_FLOAT;
        int closest_centroid = 0;

        for (int centroid = 0; centroid < K; centroid++)
        {
            // float distance = squareDistance(&points[point_idx], &centroids[centroid * n], n);

            float distance = 0;
            for (int i = 0; i < n; i++)
            {
                distance += 
                    (points[point_idx + i] - s_centroids[centroid * n + i]) * 
                    (points[point_idx + i] - s_centroids[centroid * n + i]);
            }

            if (distance < min_distance)
            {
                min_distance = distance;
                closest_centroid = centroid;
            }
        }

        assignments[global_idx] = closest_centroid;
    }

    __global__ void prepareData(float* points, int* assignments, float* centroids_data, int* cluster_sizes_data, int N, int n, int K)
    {
        const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int global_point_idx = global_idx * n;
        const int global_data_idx = global_idx * K * n;
        const int global_assignment_idx = global_idx * K;

        if (global_idx >= N)
        {
            return;
        }

        int assigned_centroid = assignments[global_idx];
        for (int centroid = 0; centroid < K; centroid++)
        {
            int belongs_to_cluster = assigned_centroid == centroid;
            cluster_sizes_data[global_assignment_idx + centroid] = belongs_to_cluster;
            for (int dimension = 0; dimension < n; dimension++)
            {
                centroids_data[global_data_idx + centroid * n + dimension] = points[global_point_idx + dimension] * belongs_to_cluster;
            }
        }
    }

    __device__ void warpReduce(
        volatile float* s_centroids_data, volatile int* s_cluster_sizes_data, 
        const int local_idx, const int global_idx, 
        const int data_size, const int K, const int term_size, const int memory_offset)
    {
        const int local_centroid_idx = local_idx * term_size;
        const int local_cluster_size_idx = local_idx * K;

        // loop unrolling for better performence (as per official guide)

        // S = 32
        if (global_idx + 32 >= data_size)
        {
            return;
        }
        for (int i = 0; i < K; i++)
        {
            s_cluster_sizes_data[memory_offset + local_cluster_size_idx + i] +=
                s_cluster_sizes_data[memory_offset + local_cluster_size_idx + 32 * K + i];
        }
        for (int i = 0; i < term_size; i++)
        {
            s_centroids_data[local_centroid_idx + i] +=
                s_centroids_data[local_centroid_idx + 32 * term_size + i];
        }

        // S = 16
        if (global_idx + 16 >= data_size)
        {
            return;
        }
        for (int i = 0; i < K; i++)
        {
            s_cluster_sizes_data[memory_offset + local_cluster_size_idx + i] +=
                s_cluster_sizes_data[memory_offset + local_cluster_size_idx + 16 * K + i];
        }
        for (int i = 0; i < term_size; i++)
        {
            s_centroids_data[local_centroid_idx + i] +=
                s_centroids_data[local_centroid_idx + 16 * term_size + i];
        }

        // S = 8
        if (global_idx + 8 >= data_size)
        {
            return;
        }
        for (int i = 0; i < K; i++)
        {
            s_cluster_sizes_data[memory_offset + local_cluster_size_idx + i] +=
                s_cluster_sizes_data[memory_offset + local_cluster_size_idx + 8 * K + i];
        }
        for (int i = 0; i < term_size; i++)
        {
            s_centroids_data[local_centroid_idx + i] +=
                s_centroids_data[local_centroid_idx + 8 * term_size + i];
        }

        // S = 4
        if (global_idx + 4 >= data_size)
        {
            return;
        }
        for (int i = 0; i < K; i++)
        {
            s_cluster_sizes_data[memory_offset + local_cluster_size_idx + i] +=
                s_cluster_sizes_data[memory_offset + local_cluster_size_idx + 4 * K + i];
        }
        for (int i = 0; i < term_size; i++)
        {
            s_centroids_data[local_centroid_idx + i] +=
                s_centroids_data[local_centroid_idx + 4 * term_size + i];
        }

        // S = 2
        if (global_idx + 2 >= data_size)
        {
            return;
        }
        for (int i = 0; i < K; i++)
        {
            s_cluster_sizes_data[memory_offset + local_cluster_size_idx + i] +=
                s_cluster_sizes_data[memory_offset + local_cluster_size_idx + 2 * K + i];
        }
        for (int i = 0; i < term_size; i++)
        {
            s_centroids_data[local_centroid_idx + i] +=
                s_centroids_data[local_centroid_idx + 2 * term_size + i];
        }

        // S = 1
        if (global_idx + 1 >= data_size)
        {
            return;
        }
        for (int i = 0; i < K; i++)
        {
            s_cluster_sizes_data[memory_offset + local_cluster_size_idx + i] +=
                s_cluster_sizes_data[memory_offset + local_cluster_size_idx + 1 * K + i];
        }
        for (int i = 0; i < term_size; i++)
        {
            s_centroids_data[local_centroid_idx + i] +=
                s_centroids_data[local_centroid_idx + 1 * term_size + i];
        }
    }

    __global__ void sumData(float* centroids_data, int* cluster_sizes_data, int data_size, int n, int K)
    {
        int term_size = K * n;

        // blockDim.x*2 as suggested in the official parallel reduction guide
        // because in the first iteration half the threads are idle
        // so we will perform the first addition while copying the data
        const int global_idx = blockIdx.x * (blockDim.x*2) + threadIdx.x;
        const int local_idx = threadIdx.x;

        const int global_centroid_idx = global_idx * term_size;
        const int local_centroid_idx = local_idx * term_size;

        const int global_cluster_size_idx = global_idx * K;
        const int local_cluster_size_idx = local_idx * K;

        if (global_idx >= data_size)
        {
            return;
        }

        // NOTE
        // To sum all the coordinates of points in a given cluster we are going to use parallel reduction
        // This process sums a partition of data assigned to a given block to a single value
        // and then recursively repeat the process by launching the kernel again

        // However we need do not want to sum everything, we want to have K independent sums for each cluster.
        // To achieve this we will expand each point to be of size K * n and set 0 to every cell except
        // the one corresponding to the cluster the point belongs to.
        // This way we can sum everything up normally and in the end the resulting array of K * n will have sums
        // of coordinates for all clusters grouped together and easily seperable.

        extern __shared__ float s_centroids_data[]; // size if THREADS_PER_BLOCK * K * n + THREADS_PER_BLOCK * K;
        extern __shared__ int s_cluster_sizes_data[];
        int memory_offset = THREADS_PER_BLOCK * K * n;
        
        // copy from global to shared, perform initial addition if possible
        for (int i = 0; i < K; i++)
        {
            s_cluster_sizes_data[memory_offset + local_cluster_size_idx + i] = 
                cluster_sizes_data[global_cluster_size_idx + i];
            if (global_idx + blockDim.x < data_size)
            {
                s_cluster_sizes_data[memory_offset + local_cluster_size_idx + i] += 
                    cluster_sizes_data[global_cluster_size_idx + blockDim.x * K + i];
            }
        }

        // copy from global to shared, perform initial addition if possible
        for (int i = 0; i < term_size; i++)
        {
            s_centroids_data[local_centroid_idx + i] = centroids_data[global_centroid_idx + i];
            if (global_idx + blockDim.x < data_size)
            {
                s_centroids_data[local_centroid_idx + i] += centroids_data[global_centroid_idx + blockDim.x * term_size + i];
            }
        }

        __syncthreads();

        for (unsigned int s = blockDim.x/2; s > 32; s>>=1)
        {
            if (local_idx < s)
            {
                // check if not outside of range
                if (global_idx + s >= data_size)
                {
                    break;
                }

                for (int i = 0; i < K; i++)
                {
                    s_cluster_sizes_data[memory_offset + local_cluster_size_idx + i] += 
                        s_cluster_sizes_data[memory_offset + local_cluster_size_idx + s * K + i];
                }
                for (int i = 0; i < term_size; i++)
                {
                    s_centroids_data[local_centroid_idx + i] += 
                        s_centroids_data[local_centroid_idx + s * term_size + i];
                }
            }
            __syncthreads();
        }

        if (local_idx < 32)
        {
            warpReduce(s_centroids_data, s_cluster_sizes_data, local_idx, global_idx, data_size, K, term_size, memory_offset);
        }

        // copy to global
        if (local_idx == 0)
        {
            for (int i = 0; i < K; i++)
            {
                cluster_sizes_data[blockIdx.x * K + i] = s_cluster_sizes_data[memory_offset + i];
            }
            for (int i = 0; i < term_size; i++)
            {
                centroids_data[blockIdx.x * term_size + i] = s_centroids_data[i];
            }
        }
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
                int assigned_centroid = s_assignments[point];
                b_cluster_sizes[assigned_centroid]++;

                // add point coordinates to the relevant sum
                for (int dimension = 0; dimension < n; dimension++)
                {
                    b_coordinates_sums[assigned_centroid * n + dimension] += s_points[memory_offset + point * n + dimension];
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