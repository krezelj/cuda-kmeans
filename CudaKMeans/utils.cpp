#include <cstdlib>
#include <random>
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <numeric>

#include "utils.h"

float* generateGaussianCluster(int N, int n, float* mean, float variance);

float* sampleCentroids(float* points, int N, int n, int K)
{
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 1);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    float* centroids = new float[K * n];
    for (int i = 0; i < K; i++)
    {
        std::memcpy(&centroids[i * n], &points[indices[i] * n], n * sizeof(float));
    }
    return centroids;
}

float* generatePoints(int N, int n)
{
    float* points = new float[N * n];
    for (int i = 0; i < N * n; i++)
    {
        points[i] = rand() / static_cast<float>(RAND_MAX) * 10;
    }

    return points;
}

float* generateGaussianPoints(int points_per_cluster, int n, int K, float* means, float* variances)
{
    int N = points_per_cluster * K;
    float* points = new float[N * n];
    float* current_mean = new float[n];
    float* output = new float[points_per_cluster * n];
    float current_variance;
    for (int i = 0; i < K; i++)
    {
        std::memcpy(current_mean, &means[i * n], n * sizeof(float));
        current_variance = variances[i];

        output = generateGaussianCluster(points_per_cluster, n, current_mean, current_variance);
        std::memcpy(&points[points_per_cluster * i * n], output, points_per_cluster * n * sizeof(float));
    }

    delete[] current_mean;
    delete[] output;

    return points;
}

float* generateGaussianCluster(int N, int n, float* mean, float variance)
{
    float* cluster_points = new float[N * n];
    
    std::default_random_engine generator;
    for (int dimension = 0; dimension < n; dimension++)
    {
        std::normal_distribution<float> distribution(mean[dimension], variance);
        for (int i = 0; i < N; i++)
        {
            float coordinate = distribution(generator);
            cluster_points[i * n + dimension] = coordinate;
        }
    }
    return cluster_points;
}

void writeCentroidToFile(float* centroids, int n, int K, std::string file_path)
{
    std::ofstream result_file;
    result_file.open(file_path);

    for (int i = 0; i < K; i++)
    {
        for (int dimension = 0; dimension < n; dimension++)
        {
            result_file << centroids[i * n + dimension] << ",";
        }
        result_file << i << "\n";
    }
    result_file.close();
}

void writePointsToFile(float* points, int N, int n, std::string file_path)
{
    std::ofstream result_file;
    result_file.open(file_path);

    for (int i = 0; i < N; i++)
    {
        for (int dimension = 0; dimension < n - 1; dimension++)
        {
            result_file << points[i * n + dimension] << ",";
        }
        result_file << points[i * n + n - 1] << "\n";
    }
    result_file.close();
}

void writeClustersToFile(float* points, int* assignments, int N, int n, std::string file_path)
{
    std::ofstream result_file;
    result_file.open(file_path);

    for (int i = 0; i < N; i++)
    {
        for (int dimension = 0; dimension < n; dimension++)
        {
            result_file << points[i * n + dimension] << ",";
        }
        result_file << assignments[i] << "\n";
    }
    result_file.close();
}