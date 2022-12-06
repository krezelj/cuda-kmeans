﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "cpukmeans.h"
#include "gpukmeans.cuh"
#include "utils.h"


void display_centroids(float* centroids, int K, int n)
{
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << centroids[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}


int main()
{
    srand(time(0));

    const int N = 1000000, n = 3, K = 8;

    float* points = generatePoints(N, n);
    float* centroids;

    centroids = CPU::cpuKMeans(points, N, n, K, 100);
    std::cout << "CPU" << std::endl;
    display_centroids(centroids, K, n);

    centroids = GPU::gpuKMeans(points, N, n, K, 100);
    std::cout << "GPU" << std::endl;
    display_centroids(centroids, K, n);

    return 0;
}
