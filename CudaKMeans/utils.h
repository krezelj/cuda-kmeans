#pragma once

#define MAX_FLOAT 1000
#define THREADS_PER_BLOCK 32

float* sampleCentroids(int N, int n, int K);

float* generatePoints(int N, int n);

float* generateGaussianPoints(int N, int n, int K);
