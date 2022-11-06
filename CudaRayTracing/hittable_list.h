#pragma once

#include "hittable.h"

#include <vector>
#include <memory>

class hittable_list : public hittable {
public:
	XPU hittable_list() {}
	XPU hittable_list(hittable** obj_list, int n) :
		objects(obj_list), size(n) {}

	//XPU void clear() { objects.clear(); }
	//XPU void add(std::shared_ptr<hittable> obj) { objects.push_back(obj); }

	XPU virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
	GPU virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

public:
	hittable** objects;
	int size;
};

XPU inline bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	hit_record temp_rec;
	bool hit_anything = false;
	float closest_so_far = t_max;

	for (int i = 0; i < size; i++) {
		if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}

	return hit_anything;
}

GPU inline bool hittable_list::bounding_box(float time0, float time1, aabb& output_box) const {
	if (size == 0)
		return false;

	aabb temp_box;
	bool first_box = true;

	for (int i = 0; i < size; i++) {
		if (!objects[i]->bounding_box(time0, time1, temp_box))
			return false;
		output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
		first_box = false;
	}

	return true;
}
