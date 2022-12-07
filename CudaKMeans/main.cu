#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>

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

void benchmark_test(int N, int n, int K)
{
    std::chrono::steady_clock::time_point start, stop;
    std::chrono::milliseconds duration;

    const int max_iterations = 50;

    float* points = generatePoints(N, n);
    std::cout << "N: " << N << "\nn: " << n << "\nK: " << K << std::endl;

    // cpu
    std::cout << "CPU... ";

    start = std::chrono::high_resolution_clock::now();
    CPU::cpuKMeans(points, N, n, K, max_iterations);
    stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << "ms" << std::endl;

    // gpu method 1
    std::cout << "GPU (Method 1)... ";

    start = std::chrono::high_resolution_clock::now();
    GPU::gpuKMeans(points, N, n, K, max_iterations, GPU::SIMPLIFIED_PARALLEL_REDUCTION);
    stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << "ms" << std::endl;

    // gpu method 2
    std::cout << "GPU (Method 2)... ";

    start = std::chrono::high_resolution_clock::now();
    GPU::gpuKMeans(points, N, n, K, max_iterations, GPU::PROPER_PARALLEL_REDUCTION);
    stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << "ms" << std::endl;
}

void benchmark()
{
    std::cout << "\n----- BENCHMARK 1 -----" << std::endl;
    std::cout << "Small dataset" << std::endl;
    benchmark_test(100000, 3, 6);

    std::cout << "\n----- BENCHMARK 2 -----" << std::endl;
    std::cout << "Medium dataset" << std::endl;
    benchmark_test(1000000, 3, 6);

    std::cout << "\n----- BENCHMARK 3 -----" << std::endl;
    std::cout << "Large dataset" << std::endl;
    benchmark_test(5000000, 4, 8);

    std::cout << "\n----- BENCHMARK 4 -----" << std::endl;
    std::cout << "XLarge dataset, low dimensionality, few clusters" << std::endl;
    benchmark_test(20000000, 2, 3);

    std::cout << "\n----- BENCHMARK 5 -----" << std::endl;
    std::cout << "Large dataset, low dimensionality, many clusters" << std::endl;
    benchmark_test(5000000, 2, 10);

    std::cout << "\n----- BENCHMARK 6 -----" << std::endl;
    std::cout << "Large dataset, high dimensionality, few clusters" << std::endl;
    benchmark_test(5000000, 10, 3);

    std::cout << "\n----- BENCHMARK 7 -----" << std::endl;
    std::cout << "Small dataset, high dimensionality, many clusters" << std::endl;
    benchmark_test(100000, 12, 12);
}

int main()
{
    srand(time(NULL));
    benchmark();

    return 0;
}
