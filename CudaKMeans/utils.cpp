#include <cstdlib>
#include "utils.h"

float* sampleCentroids(int N, int n, int K)
{
    // TODO Change sampling algorithm to initialise with real points
    return generatePoints(K, n);
}

float* generatePoints(int N, int n)
{
    float* points = new float[N * n];
    for (int i = 0; i < N * n; i++)
    {
        points[i] = rand() / static_cast<float>(RAND_MAX);
    }

    return points;
}

float* generateGaussianPoints(int N, int n, int K)
{
    return NULL;
}