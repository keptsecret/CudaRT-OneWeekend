#pragma once

#include "util.h"
#include "vec3.h"

#include <iostream>

void write_color(std::ostream& out, const color& pixel_color, const int samples_per_pixel) {
	float r = pixel_color.x();
	float g = pixel_color.y();
	float b = pixel_color.z();

	// Divide intensity by number of samples and correct with "gamma 2"
	const float scale = 1.0f / samples_per_pixel;
	r = std::sqrt(r * scale);
	g = std::sqrt(g * scale);
	b = std::sqrt(b * scale);

	// Write the translated [0,255] value of each color component
	out << static_cast<int>(256 * clamp(r, 0, 0.999f)) << ' '
		<< static_cast<int>(256 * clamp(g, 0, 0.999f)) << ' '
		<< static_cast<int>(256 * clamp(b, 0, 0.999f)) << '\n';
}

void write_color(unsigned char* pixel, const color& pixel_color, const int samples_per_pixel) {
	float r = pixel_color.x();
	float g = pixel_color.y();
	float b = pixel_color.z();

	// Divide intensity by number of samples and correct with "gamma 2"
	const float scale = 1.0f / samples_per_pixel;
	r = std::sqrt(r * scale);
	g = std::sqrt(g * scale);
	b = std::sqrt(b * scale);

	// Write the translated [0,255] value of each color component
	pixel[0] = static_cast<unsigned char>(256 * clamp(r, 0, 0.999f));
	pixel[1] = static_cast<unsigned char>(256 * clamp(g, 0, 0.999f));
	pixel[2] = static_cast<unsigned char>(256 * clamp(b, 0, 0.999f));
}