#pragma once

#include "ray.h"
#include "util.h"
#include "vec3.h"

class camera {
public:
	GPU camera() {
        lower_left_corner = vec3(-2.0, -1.0, -1.0);
        horizontal = vec3(4.0, 0.0, 0.0);
        vertical = vec3(0.0, 2.0, 0.0);
        origin = vec3(0.0, 0.0, 0.0);

		u = v = w = vec3();
		lens_radius = 0;
		time0 = time1 = 0;
    }

	GPU camera(point3 lookfrom, point3 lookat, point3 vup, float fov, float aspect_ratio, float aperture, float focus_dist, float t0 = 0, float t1 = 0) :
			origin(lookfrom),
			lens_radius(aperture / 2),
			time0(t0),
			time1(t1) {
		float theta = degrees_to_radians(fov);
		float h = tan(theta / 2.0f);
		float viewport_width = 2.0f * h;
		float viewport_height = viewport_width / aspect_ratio;
		float focal_length = 1.0f;

		w = cu_unit_vector(lookfrom - lookat);
		u = cu_unit_vector(cross(vup, w));
		v = cu_unit_vector(cross(w, u));

		horizontal = focus_dist * viewport_width * u;
		vertical = focus_dist * viewport_height * v;
		lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;
	}

	GPU ray get_ray(float s, float t, curandState* local_rand) const {
		vec3 rand = lens_radius * cu_random_in_unit_disk(local_rand);
		vec3 offset = u * rand.x() + v * rand.y();

		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset, cu_random_float(time0, time1, local_rand));
	}

private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;

	vec3 u, v, w;
	float lens_radius;
	float time0, time1;
};
