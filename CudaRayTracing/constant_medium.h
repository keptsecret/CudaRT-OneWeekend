#pragma once

#include "hittable.h"
#include "material.h"
#include "texture.h"

class constant_medium : public hittable {
public:
	// TODO: check this
	GPU constant_medium(hittable* b, float d, cu_texture* a) :
			boundary(b), neg_inv_density(-1 / d), phase_function(&isotropic(a)) {}

	//GPU constant_medium(hittable* b, float d, color c) :
	//		boundary(b), neg_inv_density(-1 / d), phase_function(&isotropic(c)) {}

	GPU virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

	GPU virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		return boundary->bounding_box(time0, time1, output_box);
	}

public:
	hittable* boundary;
	material* phase_function;
	float neg_inv_density;
};

GPU inline bool constant_medium::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	const bool enable_debug = false;
	const bool debug = enable_debug && random_float() < 0.00001f;

	hit_record rec1, rec2;

	if (!boundary->hit(r, -infinity, infinity, rec1))
		return false;

	if (!boundary->hit(r, rec1.t + 0.0001f, infinity, rec2))
		return false;

	if (debug)
		std::cerr << "\nt_min=" << rec1.t << ", t_max=" << rec2.t << '\n';

	if (rec1.t < t_min)
		rec1.t = t_min;
	if (rec2.t > t_max)
		rec2.t = t_max;

	if (rec1.t >= rec2.t)
		return false;

	if (rec1.t < 0)
		rec1.t = 0;

	const float ray_length = r.direction().length();
	const float distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
	const float hit_distance = neg_inv_density * log(random_float());

	if (hit_distance > distance_inside_boundary)
		return false;

	rec.t = rec1.t + hit_distance / ray_length;
	rec.p = r.at(rec.t);

	if (debug) {
		std::cerr << "hit_distance = " << hit_distance << '\n'
				  << "rec.t = " << rec.t << '\n'
				  << "rec.p = " << rec.p << '\n';
	}

	rec.normal = vec3(1, 0, 0);		// arbitrary normal values
	rec.front_face = true;
	rec.material_ptr = phase_function;

	return true;
}
