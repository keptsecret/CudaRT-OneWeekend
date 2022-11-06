#pragma once

#include "aarect.h"
#include "hittable_list.h"

class cube : public hittable {
public:
	cube() {}
	cube(const point3& p0, const point3& p1, std::shared_ptr<material> mat);

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
	virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		output_box = aabb(box_min, box_max);
		return true;
	}

public:
	point3 box_min;
	point3 box_max;
	hittable_list faces;
};

inline cube::cube(const point3& p0, const point3& p1, std::shared_ptr<material> mat) {
	box_min = p0;
	box_max = p1;

	faces.add(std::make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), mat));
	faces.add(std::make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), mat));

	faces.add(std::make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), mat));
	faces.add(std::make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), mat));

	faces.add(std::make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), mat));
	faces.add(std::make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), mat));
}

inline bool cube::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	return faces.hit(r, t_min, t_max, rec);
}

