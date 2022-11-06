#pragma once

#include "hittable.h"
#include "onb.h"
#include "ray.h"
#include "texture.h"
#include "util.h"

struct hit_record;

class material {
public:
	GPU virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, float& pdf, curandState* local_rand) const {
		return false;
	}

	GPU virtual float scattering_pdf(const ray& r_in, const hit_record& rec, ray& scattered) const {
		return 0;
	}

	GPU virtual color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p) const {
		return color(0, 0, 0);
	}
};

class lambertian : public material {
public:
	//XPU lambertian(const color& a) :
	//		albedo(std::make_shared<solid_color>(a)) {}

	GPU lambertian(cu_texture* a) :
			albedo(a) {}

	GPU virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, float& pdf, curandState* local_rand) const override {
		onb uvw;
		uvw.build_from_w(rec.normal);
		vec3 scatter_dir = uvw.local(cu_random_cosine_direction(local_rand));

		scattered = ray(rec.p, cu_unit_vector(scatter_dir), r_in.time());
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		pdf = cu_dot(uvw.w(), scattered.direction()) / pi;
		return true;
	}

	GPU float scattering_pdf(const ray& r_in, const hit_record& rec, ray& scattered) const override {
		float cosine = cu_dot(rec.normal, scattered.direction());
		return cosine < 0 ? 0 : (cosine / pi);
	}

public:
	cu_texture* albedo;
};

class metal : public material {
public:
	//XPU metal(const color& a, const float r) :
	//		albedo(std::make_shared<solid_color>(a)), roughness(r < 1 ? r : 1) {}

	GPU metal(cu_texture* a, const float r) :
			albedo(a), roughness(r < 1 ? r : 1) {}

	GPU virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, float& pdf, curandState* local_rand) const override {
		vec3 reflect_dir = reflect(cu_unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflect_dir + roughness * cu_random_in_unit_sphere(local_rand), r_in.time());
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return cu_dot(scattered.direction(), rec.normal) > 0.0f;
	}

public:
	cu_texture* albedo;
	float roughness;
};

class dielectric : public material {
public:
	GPU dielectric(float ior) :
			ir(ior) {}

	GPU virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, float& pdf, curandState* local_rand) const override {
		attenuation = color(1, 1, 1);
		float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;

		vec3 unit_dir = cu_unit_vector(r_in.direction());
		float cos_theta = fmin(dot(-unit_dir, rec.normal), 1.0f);
		float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

		bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
		vec3 direction;

		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > cu_random_float(local_rand)) {
			// account for total internal reflection as well, when ray intercepts surface at almost perp angles
			direction = reflect(unit_dir, rec.normal);
		} else {
			direction = refract(unit_dir, rec.normal, refraction_ratio);
		}

		scattered = ray(rec.p, direction, r_in.time());
		return true;
	}

public:
	float ir;

private:
	GPU static float reflectance(float cosine, float ref_idx) {
		// Use Schlick's approximation for reflectance (varying reflectivity by angle)
		float r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow(1 - cosine, 5.0f);
	}
};

class diffuse_light : public material {
public:
	GPU diffuse_light(cu_texture* a) :
			emit(a) {}

	//XPU diffuse_light(color c) :
	//		emit(std::make_shared<solid_color>(c)) {}

	GPU virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, float& pdf, curandState* local_rand) const override {
		return false;
	}

	GPU virtual color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p) const override {
		if (rec.front_face)
			return emit->value(u, v, p);
		return color(0, 0, 0);
	}

public:
	cu_texture* emit;
};

class isotropic : public material {
public:
	//XPU isotropic(color c) : albedo(std::make_shared<solid_color>(c)) {}

	GPU isotropic(cu_texture* a) : albedo(a) {}

	GPU virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, float& pdf, curandState* local_rand) const override {
		scattered = ray(rec.p, cu_random_in_unit_sphere(local_rand), r_in.time());
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}

public:
	cu_texture* albedo;
};