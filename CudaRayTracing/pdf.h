#pragma once
#include "onb.h"
#include "vec3.h"

class pdf {
public:
	GPU virtual ~pdf() {}

	GPU virtual float value(const vec3& direction) const = 0;
	GPU virtual vec3 generate(curandState* local_rand) const = 0;
};

class cosine_pdf : public pdf {
public:
	GPU cosine_pdf(const vec3& w) { uvw.build_from_w(w); }

	GPU virtual float value(const vec3& direction) const override {
		float cosine = cu_dot(cu_unit_vector(direction), uvw.w());
		return cosine < 0 ? 0 : (cosine / pi);
	}

	GPU virtual vec3 generate(curandState* local_rand) const override {
		return uvw.local(cu_random_cosine_direction(local_rand));
	}

public:
	onb uvw;
};

class hittable_pdf : public pdf {
public:
	GPU hittable_pdf(const point3& origin, hittable* p) :
			o(origin), ptr(p) {}

	GPU virtual float value(const vec3& direction) const override {
		return ptr->pdf_value(o, direction);
	}

	GPU virtual vec3 generate(curandState* local_rand) const override {
		return ptr->random(o, local_rand);
	}

public:
	point3 o;
	hittable* ptr;
};

class mixture_pdf : public pdf {
public:
	GPU mixture_pdf(pdf* p0, pdf* p1) {
		p[0] = p0;
		p[1] = p1;
	}

	GPU virtual float value(const vec3& direction) const override {
		return 0.5f * p[0]->value(direction) + 0.5f * p[1]->value(direction);
	}

	GPU virtual vec3 generate(curandState* local_rand) const override {
		if (cu_random_float(local_rand) < 0.5f)
			return p[0]->generate(local_rand);
		else
			return p[1]->generate(local_rand);
	}

public:
	pdf* p[2];
};
