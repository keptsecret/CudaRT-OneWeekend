#pragma once
#include "ray.h"
#include "vec3.h"

class aabb {
public:
	XPU aabb() {}
	XPU aabb(const point3& a, const point3& b) :
			minimum(a), maximum(b) {}

	XPU point3 min() const { return minimum; }
	XPU point3 max() const { return maximum; }

	XPU bool hit(const ray& r, float t_min, float t_max) const;

public:
	point3 minimum;
	point3 maximum;
};

XPU inline bool aabb::hit(const ray& r, float t_min, float t_max) const {
	for (int idx = 0; idx < 3; idx++) {
		float inv_d = 1.0f / r.direction()[idx];
		float t0 = (min()[idx] - r.origin()[idx]) * inv_d;
		float t1 = (max()[idx] - r.origin()[idx]) * inv_d;
		if (inv_d < 0.0f)
			std::swap(t0, t1);
		t_min = t0 > t_min ? t0 : t_min;
		t_max = t1 < t_max ? t1 : t_max;
		if (t_min >= t_max)
			return false;
	}
	return true;
}

XPU inline aabb surrounding_box(aabb& box0, aabb& box1) {
	point3 mini(fmin(box0.min().x(), box1.min().x()),
			fmin(box0.min().y(), box1.min().y()),
			fmin(box0.min().z(), box1.min().z()));

	point3 maxi(fmax(box0.max().x(), box1.max().x()),
			fmax(box0.max().y(), box1.max().y()),
			fmax(box0.max().z(), box1.max().z()));

	return aabb(mini, maxi);
}
