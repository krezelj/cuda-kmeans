#pragma once

#define MAX_FLOAT 1000
#define THREADS_PER_BLOCK 64

float* sampleCentroids(int N, int n, int K);

float* generatePoints(int N, int n);
