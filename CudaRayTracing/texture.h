#pragma once
#include "color.h"
#include "perlin.h"
#include "stb_image.h"

class cu_texture {
public:
	XPU virtual color value(float u, float v, const point3& p) const = 0;
};

class solid_color : public cu_texture {
public:
	XPU solid_color() {}
	XPU solid_color(color c) :
			color_value(c) {}

	XPU solid_color(float r, float g, float b) :
			solid_color(color(r, g, b)) {}

	XPU virtual color value(float u, float v, const point3& p) const override {
		return color_value;
	}

private:
	color color_value;
};

class checker_texture : public cu_texture {
public:
	XPU checker_texture() {}

	XPU checker_texture(cu_texture* e, cu_texture* o) :
		odd(o), even(e) {}

	//XPU checker_texture(color c1, color c2) :				///> don't know how to replace makr_shared
	//		odd(solid_color(c2)), even(solid_color(c1)) {}

	XPU virtual color value(float u, float v, const point3& p) const override {
		float sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
		if (sines < 0)
			return odd->value(u, v, p);
		else
			return even->value(u, v, p);
	}

public:
	cu_texture* odd;
	cu_texture* even;
};

class noise_texture : public cu_texture {
public:
	XPU noise_texture() {}

	XPU noise_texture(const float sc) :
			scale(sc) {}

	XPU virtual color value(float u, float v, const point3& p) const override {
		return color(1, 1, 1) * 0.5 * (1.0f + sin(scale * p.z() + 10 * noise.turbulence(scale * p)));
	}

private:
	perlin noise;
	float scale;
};

class image_texture : public cu_texture {
public:
	const static int bytes_per_pixel = 3;

	XPU image_texture() :
			data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

	XPU image_texture(const char* filename) {
		auto components_per_pixel = bytes_per_pixel;

		data = stbi_load(filename, &width, &height, &components_per_pixel, components_per_pixel);

		if (!data) {
			std::cerr << "ERROR::Image_texture: Could not load image texture file " << filename << ".\n";
			width = height = 0;
		}

		bytes_per_scanline = bytes_per_pixel * width;
	}

	XPU ~image_texture() {
		delete data;
	}

	XPU virtual color value(float u, float v, const point3& p) const override {
		// return solid cyan if no image texture found
		if (data == nullptr)
			return color(0, 1, 1);

		// clamp input texture coordinates to [0, 1] x [1, 0]
		u = cu_clamp(u, 0.0f, 1.0f);
		v = 1.0f - cu_clamp(v, 0.0f, 1.0f);	// flip vertical coords

		auto i = static_cast<int>(u * width);
		auto j = static_cast<int>(v * height);

		// clamp integer mapping since actual coordinates should be < 1.0
		if (i >= width)
			i = width - 1;
		if (j >= height)
			j = height - 1;

		const float color_scale = 1.0f / 255.0f;
		unsigned char* pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;

		return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
	}

private:
	unsigned char* data;
	int width, height;
	int bytes_per_scanline;
};