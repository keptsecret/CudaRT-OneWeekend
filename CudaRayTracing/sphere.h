#pragma once

#include "vec3.h"
#include "hittable.h"
#include "onb.h"

class sphere : public hittable {
public:
	XPU sphere() {}

	//XPU sphere(point3 c, float r) :
	//		center(c), radius(r) {}
	// 
	XPU sphere(point3 c, float r, material* m) :
			center(c), radius(r), material_ptr(m) {}

	XPU virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
	GPU virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;
	GPU virtual float pdf_value(const point3& o, const vec3& v) const override;
	GPU virtual vec3 random(const vec3& o, curandState* local_rand) const override;

public:
	point3 center;
	float radius;
	material* material_ptr;

private:
	XPU static void get_sphere_uv(const point3& p, float& u, float& v) {
		// p: a given point on the sphere of radius one, centered at the origin.
		// u: returned value [0,1] of angle around the Y axis from X=-1.
		// v: returned value [0,1] of angle from Y=-1 to Y=+1.
		//     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
		//     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
		//     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

		float theta = acos(-p.y());
		float phi = atan2(-p.z(), p.x()) + pi;

		u = phi / (2 * pi);
		v = theta / pi;
	}
};

XPU inline bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	// implements simplified quadratic equation for vector dotted with itself
	vec3 oc = r.origin() - center;
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
	vec3 outward_normal = (rec.p - center) / radius;
	rec.set_face_normal(r, outward_normal);
	//get_sphere_uv(outward_normal, rec.u, rec.v);
	rec.material_ptr = material_ptr;

	return true;
}

GPU inline bool sphere::bounding_box(float time0, float time1, aabb& output_box) const {
	output_box = aabb(
			center - vec3(radius, radius, radius),
			center + vec3(radius, radius, radius));
	return true;
}

GPU inline float sphere::pdf_value(const point3 &o, const vec3 &v) const {
	hit_record rec;
	if (!this->hit(ray(o, v), 0.001f, infinity, rec)) {
		return 0;
	}

	auto cos_theta_max = sqrt(1 - radius * radius / (center - o).length_squared());
	auto solid_angle = 2 * pi * (1 - cos_theta_max);

    return  1 / solid_angle;
}


GPU inline vec3 sphere::random(const point3& o, curandState* local_rand) const {
     vec3 direction = center - o;
     auto distance_squared = direction.length_squared();
     onb uvw;
     uvw.build_from_w(direction);
     return uvw.local(cu_random_to_sphere(radius, distance_squared, local_rand));
}
