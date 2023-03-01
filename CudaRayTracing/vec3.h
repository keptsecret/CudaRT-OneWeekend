#pragma once

#include "util.h"

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3 {
public:
	XPU vec3() :
			e{ 0.0f, 0.0f, 0.0f } {}
	XPU vec3(float e0, float e1, float e2) :
			e{ e0, e1, e2 } {}

	// Setters, getters and operators
	XPU inline float x() const { return e[0]; }
	XPU inline float y() const { return e[1]; }
	XPU inline float z() const { return e[2]; }

	XPU inline vec3 operator-() const { return vec3{ -e[0], -e[1], -e[2] }; }
	XPU inline float operator[](int i) const { return e[i]; }
	XPU inline float& operator[](int i) { return e[i]; }

	XPU inline vec3& operator+=(const vec3& v) {
		e[0] += v[0];
		e[1] += v[1];
		e[2] += v[2];
		return *this;
	}

	XPU inline vec3& operator*=(const vec3& v) {
		e[0] *= v[0];
		e[1] *= v[1];
		e[2] *= v[2];
		return *this;
	}

	XPU inline vec3& operator*=(const float t) {
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}

	XPU inline vec3& operator/=(const float t) {
		return *this *= 1 / t;
	}

	XPU inline float length_squared() const {
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	XPU inline float length() const {
		return std::sqrt(length_squared());
	}

	inline static vec3 random() {
		return vec3{ random_float(), random_float(), random_float() };
	}

	inline static vec3 random(float min, float max) {
		return vec3{ random_float(min, max), random_float(min, max), random_float(min, max) };
	}

	GPU inline static vec3 cu_random(curandState* local_rand) {
		return vec3{ curand_uniform(local_rand), curand_uniform(local_rand), curand_uniform(local_rand) };
	}

	GPU inline static vec3 cu_random(float min, float max, curandState* local_rand) {
		return vec3{ cu_random_float(min, max, local_rand), cu_random_float(min, max, local_rand), cu_random_float(min, max, local_rand) };
	}

	XPU bool near_zero() const {
		// Returns true if vector is close to 0 in all directions
		const float eps = 1e-8f;
		return (fabs(e[0]) < eps) && (fabs(e[1]) < eps) && (fabs(e[2]) < eps);
	}

public:
	float e[3];
};


// Utility functions
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

XPU inline vec3 operator+(const vec3& u, const vec3& v) {
	return vec3{ u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2] };
}

XPU inline vec3 operator-(const vec3& u, const vec3& v) {
	return vec3{ u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2] };
}

XPU inline vec3 operator*(const vec3& u, const vec3& v) {
	return vec3{ u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2] };
}

XPU inline vec3 operator*(float t, const vec3& v) {
	return vec3{ t * v.e[0], t * v.e[1], t * v.e[2] };
}

XPU inline vec3 operator*(const vec3& v, float t) {
	return t * v;
}

XPU inline vec3 operator/(vec3 v, float t) {
	return (1 / t) * v;
}

XPU inline float dot(const vec3& u, const vec3& v) {
	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

GPU inline float cu_dot(const vec3& u, const vec3& v) {
	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

XPU inline vec3 cross(const vec3& u, const vec3& v) {
	return vec3{ u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0] };
}

GPU inline vec3 cu_cross(const vec3& u, const vec3& v) {
	return vec3{ u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0] };
}

XPU inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}

GPU inline vec3 cu_unit_vector(const vec3& v) {
	float n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	float inv_n = 1/n;
	return vec3(v[0] * inv_n, v[1] * inv_n, v[2] * inv_n);
}

// Type aliases of vec3
using point3 = vec3;
using color = vec3;

#ifndef USE_CUDA

inline vec3 random_in_unit_disk() {
	while (true) {
		vec3 p = vec3(random_float(-1, 1), random_float(-1, 1), 0.0f);
		if (p.length_squared() >= 1)
			continue;
		return p;
	}
}

inline vec3 random_in_unit_sphere() {
	while (true) {
		vec3 p = vec3::random(-1, 1);
		if (p.length_squared() >= 1)
			continue;
		return p;
	}
}

inline vec3 random_unit_vector() {
	return unit_vector(random_in_unit_sphere());
}

inline vec3 random_cosine_direction() {
	float r1 = random_float();
	float r2 = random_float();
	float z = sqrt(1 - r2);

	float phi = 2 * pi * r1;
	float x = cos(phi) * sqrt(r2);
	float y = sin(phi) * sqrt(r2);

	return vec3(x, y, z);
}

inline vec3 random_to_sphere(float radius, float distance_squared) {
    auto r1 = random_float();
    auto r2 = random_float();
    auto z = 1 + r2*(sqrt(1-radius*radius/distance_squared) - 1);

    auto phi = 2*pi*r1;
    auto x = cos(phi)*sqrt(1-z*z);
    auto y = sin(phi)*sqrt(1-z*z);

    return vec3(x, y, z);
}

#else

GPU inline vec3 cu_random_in_unit_disk(curandState* local_rand) {
	while (true) {
		vec3 p = vec3(curand_uniform(local_rand), curand_uniform(local_rand), 0.0f);
		if (p.length_squared() >= 1)
			continue;
		return p;
	}
}

GPU inline vec3 cu_random_in_unit_sphere(curandState* local_rand) {
	while (true) {
		vec3 p = vec3::cu_random(-1, 1, local_rand);
		if (p.length_squared() >= 1)
			continue;
		return p;
	}
}

GPU inline vec3 cu_random_unit_vector(curandState* local_rand) {
	return unit_vector(cu_random_in_unit_sphere(local_rand));
}

GPU inline vec3 cu_random_cosine_direction(curandState* local_rand) {
	float r1 = cu_random_float(local_rand);
	float r2 = cu_random_float(local_rand);
	float z = sqrt(1 - r2);

	float phi = 2 * pi * r1;
	float x = cos(phi) * sqrt(r2);
	float y = sin(phi) * sqrt(r2);

	return vec3(x, y, z);
}

GPU inline vec3 cu_random_to_sphere(float radius, float distance_squared, curandState* local_rand) {
    auto r1 = cu_random_float(local_rand);
    auto r2 = cu_random_float(local_rand);
    auto z = 1 + r2 * (sqrt(1 - radius * radius / distance_squared) - 1);

    auto phi = 2*pi*r1;
    auto x = cos(phi)*sqrt(1-z*z);
    auto y = sin(phi)*sqrt(1-z*z);

    return vec3(x, y, z);
}

GPU inline vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2 * cu_dot(v, n) * n;
}

GPU inline vec3 refract(const vec3& v, const vec3& n, float etai_over_etar) {
	float cos_theta = fmin(cu_dot(-v, n), 1.0f);
	vec3 ro_perp = etai_over_etar * (v + cos_theta * n);
	vec3 ro_para = -sqrt(fabs(1.0f - ro_perp.length_squared())) * n;
	return ro_perp + ro_para;
}

#endif // USE_CUDA