#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "cpukmeans.h"
#include "gpukmeans.cuh"
#include "utils.h"


int main()
{
    srand(time(0));

    const int N = 500000, n = 2, K = 4;

    float* points = generatePoints(N, n);
    float* centroids;

    centroids = CPU::cpuKMeans(points, N, n, K, 100);
    std::cout << "CPU" << std::endl;
    for (int i = 0; i < K; i++)
    {
        std::cout << centroids[i * n] << " " << centroids[i * n + 1] << std::endl;
    }

    centroids = GPU::gpuKMeans(points, N, n, K, 100);
    std::cout << "GPU" << std::endl;
    for (int i = 0; i < K; i++)
    {
        std::cout << centroids[i * n] << " " << centroids[i * n + 1] << std::endl;
    }

    return 0;
}
