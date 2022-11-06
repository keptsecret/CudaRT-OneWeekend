#pragma once

#include <memory>
#include <limits>
#include <random>
#include "cuda_runtime.h"
#include "curand_kernel.h"

#define USE_CUDA	// comment to compile for CPU (hopefully)

#ifndef USE_CUDA
#define XPU
#define GPU
#else
#define XPU __host__ __device__
#define GPU __device__
#endif

// Constants

constexpr float infinity = FLT_MAX;
constexpr float pi = 3.1415926535897932385f;

// Functions

XPU inline float degrees_to_radians(float degrees) {
	return degrees * pi / 180.0f;
}

inline float random_float() {
	// TODO: found that using the new generator causes some very weird lighting artifacts
	//static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
	//static std::mt19937 generator;
	//return distribution(generator);

	// Returns a random real number in [0,1)
	return rand() / (RAND_MAX + 1.0f);
}

inline float random_float(float min, float max) {
	// Returns a random real number in [min,max)
	return min + random_float() * (max - min);
}

GPU inline float cu_random_float(curandState* local_rand) {
	return curand_uniform(local_rand);
}

GPU inline float cu_random_float(float min, float max, curandState* local_rand) {
	// Returns a random real number in [min,max)
	return min + cu_random_float(local_rand) * (max - min);
}

inline int random_int(int min, int max) {
	// Returns a random integer in [min,max)
	return static_cast<int>(random_float(min, max + 1));
}

inline float clamp(float x, float min, float max) {
	if (x < min)
		return min;
	if (x > max)
		return max;
	return x;
}

XPU inline float cu_clamp(float x, float min, float max) {
	if (x < min)
		return min;
	if (x > max)
		return max;
	return x;
}