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

    float* points = generatePoints(N, n);
    std::cout << "N: " << N << "\nn: " << n << "\nK: " << K << std::endl;

    // cpu
    std::cout << "CPU... ";

    start = std::chrono::high_resolution_clock::now();
    CPU::cpuKMeans(points, N, n, K, 100);
    stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << "ms" << std::endl;

    // gpu method 1
    std::cout << "GPU (Method 1)... ";

    start = std::chrono::high_resolution_clock::now();
    GPU::gpuKMeans(points, N, n, K, 100, GPU::SIMPLIFIED_PARALLEL_REDUCTION);
    stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << "ms" << std::endl;

    // gpu method 2
    std::cout << "GPU (Method 2)... ";

    start = std::chrono::high_resolution_clock::now();
    GPU::gpuKMeans(points, N, n, K, 100, GPU::PROPER_PARALLEL_REDUCTION);
    stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << "ms" << std::endl;
}

void benchmark()
{
    std::cout << "\n----- BENCHMARK 1 -----" << std::endl;
    benchmark_test(100000, 2, 4);

    std::cout << "\n----- BENCHMARK 2 -----" << std::endl;
    benchmark_test(1000000, 3, 6);

    std::cout << "\n----- BENCHMARK 3 -----" << std::endl;
    benchmark_test(5000000, 4, 8);
}


int main()
{
    srand(time(0));

    benchmark();

    //const int N = 1000000, n = 3, K = 8;

    //float* points = generatePoints(N, n);
    //float* centroids;

    ///*centroids = CPU::cpuKMeans(points, N, n, K, 100);
    //std::cout << "CPU" << std::endl;
    //display_centroids(centroids, K, n);*/

    //auto start = std::chrono::high_resolution_clock::now();
    //centroids = GPU::gpuKMeans(points, N, n, K, 100);
    //auto stop = std::chrono::high_resolution_clock::now();

    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << duration.count() << std::endl;

    //std::cout << "GPU" << std::endl;
    //display_centroids(centroids, K, n);

    return 0;
}
