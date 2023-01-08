#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <string>
#include <conio.h>

#include "cpukmeans.h"
#include "gpukmeans.cuh"
#include "utils.h"

void display_centroids(float* centroids, int K, int n)
{
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.2f\t", centroids[i * n + j]);
        }
        printf("\n");
    }
}

void benchmark_test(int points_per_cluster, int n, int K, int benchmark_id = 0)
{
    printf("---------- BENCHMARK %d ----------\n", benchmark_id);

    int N = K * points_per_cluster;

    std::chrono::steady_clock::time_point start, stop;
    std::chrono::milliseconds duration;

    const int max_iterations = 50;

    std::string assignments_file_path = "results/assignments_" + std::to_string(benchmark_id) + ".txt";
    std::string clusters_file_path = "results/centroids_" + std::to_string(benchmark_id) + ".txt";

    printf("Start benchmark with the following parameters:\nno. points:\t%d\ndimensions:\t%d\nclusters:\t%d\n\n", N, n, K);
    printf("Results will be saved to the following files:\n");
    std::cout << "assignments:\t" << assignments_file_path << std::endl;
    std::cout << "clusters:\t" << clusters_file_path << std::endl;

    // generate clusters
    float* means = generatePoints(K, n);
    float* variances = new float[K];
    std::fill_n(variances, K, 0.5f);

    printf("\nExpected cluster centres:\n");
    for (int i = 0; i < K; i++)
    {
        for (int d = 0; d < n; d++)
        {
            printf("%.2f\t", means[i * n + d]);
        }
        printf("\n");
    }

    float* points = generateGaussianPoints(points_per_cluster, n, K, means, variances);

    float* centroids = new float[K * n];
    int* assignments = new int[N];

    start = std::chrono::high_resolution_clock::now();
    GPU::gpuKMeans(points, N, n, K, 100, GPU::SIMPLIFIED_PARALLEL_REDUCTION, centroids, assignments);
    stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "\nTime: " << duration.count() << "ms" << std::endl;

    printf("Calculated centroids:\n");
    display_centroids(centroids, K, n);

    writeClustersToFile(points, assignments, N, n, assignments_file_path);
    writeCentroidToFile(centroids, n, K, clusters_file_path);

    delete[] centroids;
    delete[] assignments;

    printf("\nBenchmark finished\n");
}

void benchmark()
{
    system("cls");
    benchmark_test(100, 2, 4, 0);
    printf("\npress any key to continue...");
    getch();

    system("cls");
    benchmark_test(2000, 3, 5, 1);
    printf("\npress any key to continue...");
    getch();

    system("cls");
    benchmark_test(2000, 2, 15, 2);
    printf("\npress any key to continue...");
    getch();

    system("cls");
    benchmark_test(15000, 3, 3, 3);
    printf("\npress any key to continue...");
    getch();

    system("cls");
    benchmark_test(1000, 10, 10, 4);
    printf("\npress any key to continue...");
    getch();

    system("cls");
    benchmark_test(2000, 3, 10, 5);
    printf("\npress any key to continue...");
    getch();
}

int main()
{
    // srand(time(NULL));
    benchmark();
    return 0;
}
