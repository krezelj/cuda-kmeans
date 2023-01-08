#pragma once
#include <string>

#define MAX_FLOAT 1000
#define THREADS_PER_BLOCK 64

float* sampleCentroids(float* points, int N, int n, int K);

float* generatePoints(int N, int n);

float* generateGaussianPoints(int points_per_cluster, int n, int K, float* means, float* variances);

void writeCentroidToFile(float* centroids, int n, int K, std::string file_path);

void writePointsToFile(float* points, int N, int n, std::string file_path);

void writeClustersToFile(float* points, int* assignments, int N, int n, std::string file_path);