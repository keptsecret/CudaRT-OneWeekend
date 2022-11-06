#pragma once

#include "vec3.h"

class ray {
public:
	XPU ray(){}
	XPU ray(const point3& origin, const point3& direction, float time = 0.0) :
			orig(origin), dir(direction), tm(time) {}

	XPU point3 origin() const { return orig; }
	XPU vec3 direction() const { return dir; }
	XPU float time() const { return tm; }

	XPU point3 at(float t) const {
		return orig + t * dir;
	}

private:
	point3 orig;
	vec3 dir;
	float tm;
};