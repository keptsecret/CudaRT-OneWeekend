#pragma once
#include "hittable.h"

class xy_rect : public hittable {
public:
	GPU xy_rect() {}

	GPU xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material* mat) :
			x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {}

	GPU virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

	GPU virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		// rect should have some dimension in z-axis as well
		output_box = aabb(point3(x0, y0, k - 0.0001f), point3(x1, y1, k + 0.0001f));
		return true;
	}

public:
	float x0, x1, y0, y1, k;
	material* mp;
};

GPU inline bool xy_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	float t = (k - r.origin().z()) / r.direction().z();
	if (t < t_min || t > t_max)
		return false;

	float x = r.origin().x() + t * r.direction().x();
	float y = r.origin().y() + t * r.direction().y();
	if (x < x0 || x > x1 || y < y0 || y > y1)
		return false;

	rec.u = (x - x0) / (x1 - x0);
	rec.v = (y - y0) / (y1 - y0);
	rec.t = t;
	vec3 outward_normal = vec3(0, 0, 1);
	rec.set_face_normal(r, outward_normal);
	rec.material_ptr = mp;
	rec.p = r.at(t);
	return true;
}

class xz_rect : public hittable {
public:
	GPU xz_rect() {}

	GPU xz_rect(float _x0, float _x1, float _z0, float _z1, float _k, material* mat) :
			x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {}

	GPU virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

	GPU virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		// rect should have some dimension in y-axis as well
		output_box = aabb(point3(x0, k - 0.0001f, z0), point3(x1, k + 0.0001f, z1));
		return true;
	}

	GPU virtual float pdf_value(const point3& o, const vec3& v) const override {
		hit_record rec;
		if (!this->hit(ray(o, v), 0.001f, infinity, rec))
			return 0;

		float area = (x1 - x0) * (z1 - z0);
		float distance_squared = rec.t * rec.t * v.length_squared();
		float cosine = fabs(dot(v, rec.normal) / v.length());

		return distance_squared / (cosine * area);
	}

	GPU virtual vec3 random(const vec3& o) const override {
		point3 random_point(random_float(x0, x1), k, random_float(z0, z1));
		return random_point - o;
	}

public:
	float x0, x1, z0, z1, k;
	material* mp;
};

GPU inline bool xz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	float t = (k - r.origin().y()) / r.direction().y();
	if (t < t_min || t > t_max)
		return false;

	float x = r.origin().x() + t * r.direction().x();
	float z = r.origin().z() + t * r.direction().z();
	if (x < x0 || x > x1 || z < z0 || z > z1)
		return false;

	rec.u = (x - x0) / (x1 - x0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	vec3 outward_normal = vec3(0, 1, 0);
	rec.set_face_normal(r, outward_normal);
	rec.material_ptr = mp;
	rec.p = r.at(t);
	return true;
}

class yz_rect : public hittable {
public:
	GPU yz_rect() {}

	GPU yz_rect(float _y0, float _y1, float _z0, float _z1, float _k, material* mat) :
			y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {}

	GPU virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

	GPU virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		// rect should have some dimension in x-axis as well
		output_box = aabb(point3(k - 0.0001f, y0, z0), point3(k + 0.0001f, y1, z1));
		return true;
	}

public:
	float y0, y1, z0, z1, k;
	material* mp;
};

GPU inline bool yz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	float t = (k - r.origin().x()) / r.direction().x();
	if (t < t_min || t > t_max)
		return false;

	float y = r.origin().y() + t * r.direction().y();
	float z = r.origin().z() + t * r.direction().z();
	if (y < y0 || y > y1 || z < z0 || z > z1)
		return false;

	rec.u = (y - y0) / (y1 - y0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	vec3 outward_normal = vec3(1, 0, 0);
	rec.set_face_normal(r, outward_normal);
	rec.material_ptr = mp;
	rec.p = r.at(t);
	return true;
}