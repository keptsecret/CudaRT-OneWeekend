#pragma once
#include "vec3.h"

/**
 * \brief Orthonomal basis class: generates orthonormal bases in 3 dimensions
 */
class onb {
public:
	GPU onb() {}

	GPU inline vec3 operator[](int i) { return axis[i]; }

	GPU vec3 u() const { return axis[0]; }
	GPU vec3 v() const { return axis[1]; }
	GPU vec3 w() const { return axis[2]; }

	GPU vec3 local(float a, float b, float c) const {
		return a * u() + b * v() + c * w();
	}

	GPU vec3 local(const vec3& a) const {
		return a.x() * u() + a.y() * v() + a.z() * w();
	}

	GPU void build_from_w(const vec3& n);

private:
	vec3 axis[3];
};

//GPU special_unit_vector()

GPU inline void onb::build_from_w(const vec3& n) {
	axis[2] = cu_unit_vector(n);
	vec3 a = (fabs(w().x()) > 0.9f) ? vec3(0, 1, 0) : vec3(1, 0, 0);
	axis[1] = cu_unit_vector(cu_cross(w(), a));
	axis[0] = cu_cross(w(), v());
}
