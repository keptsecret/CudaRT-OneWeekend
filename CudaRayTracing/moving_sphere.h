#pragma once

#include "util.h"
#include "hittable.h"

class moving_sphere : public hittable {
public:
	moving_sphere() {}
	moving_sphere(point3 c0, point3 c1, float t0, float t1, float r, std::shared_ptr<material> m) :
			center0(c0), center1(c1), time0(t0), time1(t1), radius(r), material_ptr(m) {
	}

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
	virtual bool bounding_box(float t0, float t1, aabb& output_box) const override;

	point3 center(float time) const {
		return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
	};

public:
	point3 center0, center1;
	float time0, time1;
	float radius;
	std::shared_ptr<material> material_ptr;
};

inline bool moving_sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	// implements simplified quadratic equation for vector dotted with itself
	vec3 oc = r.origin() - center(r.time());
	float a = r.direction().length_squared();
	float half_b = dot(r.direction(), oc);
	float c = oc.length_squared() - radius * radius;
	float discriminant = half_b * half_b - a * c;

	if (discriminant < 0) {
		return false;
	}

	// finds nearest root in the acceptable range
	float sqrtd = std::sqrt(discriminant);
	float root = (-half_b - sqrtd) / a;
	if (root < t_min || root > t_max) {
		root = (-half_b + sqrtd) / a;
		if (root < t_min || root > t_max) {
			return false;
		}
	}

	rec.t = root;
	rec.p = r.at(rec.t);
	vec3 outward_normal = (rec.p - center(r.time())) / radius;
	rec.set_face_normal(r, outward_normal);
	rec.material_ptr = material_ptr;

	return true;
}

inline bool moving_sphere::bounding_box(float t0, float t1, aabb& output_box) const {
	aabb box0(
			center(t0) - vec3(radius, radius, radius),
			center(t0) + vec3(radius, radius, radius));

	aabb box1(
			center(t1) - vec3(radius, radius, radius),
			center(t1) + vec3(radius, radius, radius));

	output_box = surrounding_box(box0, box1);
	return true;
}
