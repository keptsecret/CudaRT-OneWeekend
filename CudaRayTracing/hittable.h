#pragma once

#include "ray.h"
#include "aabb.h"

class material;

struct hit_record {
	point3 p;
	vec3 normal;
	material* material_ptr;
	float t = infinity;
	float u;
	float v;
	bool front_face = true;

	XPU inline void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0.0f;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable {
public:
	XPU virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
	GPU virtual bool bounding_box(float time0, float time1, aabb& output_box) const = 0;

	GPU virtual float pdf_value(const point3& o, const vec3& v) const {
		return 0.0f;
	}

	GPU virtual vec3 random(const vec3& o, curandState* local_rand) const {
		return vec3(1, 0, 0);
	}
};


/*
 * ----------------------------------------------
 * Some transform classes below
 */

class translate : public hittable {
public:
	XPU translate(hittable* p, const vec3& displacement) :
			ptr(p), offset(displacement) {}

	XPU virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
	GPU virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

public:
	hittable* ptr;
	vec3 offset;
};

XPU inline bool translate::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	ray moved_r(r.origin() - offset, r.direction(), r.time());
	if (!ptr->hit(moved_r, t_min, t_max, rec))
		return false;

	rec.p += offset;
	rec.set_face_normal(moved_r, rec.normal);

	return true;
}

GPU inline bool translate::bounding_box(float time0, float time1, aabb& output_box) const {
	if (!ptr->bounding_box(time0, time1, output_box))
		return false;

	output_box = aabb(output_box.min() + offset, output_box.max() + offset);
	return true;
}

class rotate_y : public hittable {
public:
	GPU rotate_y(hittable* p, float angle);

	XPU virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

	GPU virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		output_box = bbox;
		return has_box;
	}

public:
	hittable* ptr;
	float sin_theta;
	float cos_theta;
	bool has_box;
	aabb bbox;
};

GPU inline rotate_y::rotate_y(hittable* p, float angle) : ptr(p) {
	float radians = degrees_to_radians(angle);
	sin_theta = sin(radians);
	cos_theta = cos(radians);
	has_box = ptr->bounding_box(0, 1, bbox);

	point3 min(infinity, infinity, infinity);
	point3 max(-infinity, -infinity, -infinity);

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				float x = i * bbox.max().x() + (1 - i) * bbox.min().x();
				float y = j * bbox.max().y() + (1 - j) * bbox.min().y();
				float z = k * bbox.max().z() + (1 - k) * bbox.min().z();

				float new_x = cos_theta * x + sin_theta * z;
				float new_z = -sin_theta * x + cos_theta * z;

				vec3 tester(new_x, y, new_z);

				for (int c = 0; c < 3; c++) {
					min[c] = fmin(min[c], tester[c]);
					max[c] = fmax(max[c], tester[c]);
				}
			}
		}
	}

	bbox = aabb(min, max);
}


XPU inline bool rotate_y::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	point3 origin = r.origin();
	vec3 direction = r.direction();

	origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
	origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];

	direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
	direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

	ray rotated_r(origin, direction, r.time());

	if (!ptr->hit(rotated_r, t_min, t_max, rec))
		return false;

	point3 p = rec.p;
	vec3 normal = rec.normal;

	p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
	p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

	normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
	normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

	rec.p = p;
	rec.set_face_normal(rotated_r, normal);

	return true;
}

class flip_face : public hittable {
public:
	XPU flip_face(hittable* p) :
			ptr(p) {}

	XPU virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
		if (!ptr->hit(r, t_min, t_max, rec))
			return false;

		rec.front_face = !rec.front_face;
		return true;
	}

	GPU virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		return ptr->bounding_box(time0, time1, output_box);
	}

public:
	hittable* ptr;
};
