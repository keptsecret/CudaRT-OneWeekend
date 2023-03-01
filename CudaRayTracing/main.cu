
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <iostream>
#include <time.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"
#include "pdf.h"

// Disable pedantic warnings for this external library.
#ifdef _MSC_VER
// Microsoft Visual C++ Compiler
#pragma warning(push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Restore warning levels.
#ifdef _MSC_VER
// Microsoft Visual C++ Compiler
#pragma warning(pop)
#endif

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << ": " << cudaGetErrorString(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";

        cudaDeviceReset();
        system("pause");
        exit(result);
    }
}

/*
{
	hit_record rec;

	// exceeded the ray bounce limit, no more light gathered
	if (depth <= 0) {
		return color(0, 0, 0);
	}

	// return background color if ray hits nothing
	if (!world.hit(r, 0.001f, infinity, rec)) {
		return background;
	}

	ray scattered;
	color attenuation;
	color emitted = rec.material_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
	float pdf_val;

	if (!rec.material_ptr->scatter(r, rec, attenuation, scattered, pdf_val)) {
		return emitted;
	}

	auto p0 = std::make_shared<hittable_pdf>(rec.p, lights);
	auto p1 = std::make_shared<cosine_pdf>(rec.normal);
	mixture_pdf mixed_pdf(p0, p1);

	scattered = ray(rec.p, mixed_pdf.generate(), r.time());
	pdf_val = mixed_pdf.value(scattered.direction());

	return emitted + attenuation * rec.material_ptr->scattering_pdf(r, rec, scattered) * ray_color(scattered, background, world, lights, depth - 1) / pdf_val;
}
*/

GPU color ray_color(const ray& r, hittable** world, hittable** lights, curandState* local_rand) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);

    for(int i = 0; i < 10; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            float pdf_val;
            color emitted = rec.material_ptr->emitted(r, rec, rec.u, rec.v, rec.p);

            if(!rec.material_ptr->scatter(cur_ray, rec, attenuation, scattered, pdf_val, local_rand)) {
                return cur_attenuation * emitted;
            }
            else {
                auto p0 = hittable_pdf(rec.p, *lights);
                auto p1 = cosine_pdf(rec.normal);
                mixture_pdf mixed_pdf(&p0, &p1);

                scattered = ray(rec.p, mixed_pdf.generate(local_rand), r.time());
                pdf_val = mixed_pdf.value(scattered.direction());

                cur_attenuation *= attenuation * rec.material_ptr->scattering_pdf(r, rec, scattered) / pdf_val;
                cur_ray = scattered;
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void create_world(hittable** d_list, hittable** d_world, hittable** lights, camera** cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new sphere(vec3(0, 0, -1), 0.5f, new lambertian(new solid_color(0.8f, 0.3f, 0.3f)));
        *(d_list + 1) = new sphere(vec3(0, -100.5f, -1), 100, new lambertian(new solid_color(0.8f, 0.8f, 0.2f)));
        *(d_list + 2) = new sphere(vec3(2, 2, -1), 0.25f, new diffuse_light(new solid_color(1.0f, 1.0f, 1.0f)));
        *d_world = new hittable_list(d_list, 3);
        *lights = new sphere(vec3(2, 2, -1), 0.25f, new diffuse_light(new solid_color(1.0f, 1.0f, 1.0f)));

        point3 lookfrom(0, 0.25, 5);
        point3 lookat(0, 0.5, 0);
        vec3 vup(0, 1, 0);
        auto dist_to_focus = 10.0;
        auto aperture = 0.1;
        *cam = new camera(lookfrom, lookat, vup, 45, 12.f/8.f, aperture, (lookat - lookfrom).length());
    }
}

__global__ void render_init(int w, int h, curandState* rand) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= w || j >= h) {
        return;
    }

    int pixel = j * w + i;
    curand_init(42, pixel, 0, &rand[pixel]);
}

__global__ void render(vec3* fb, int w, int h, int samples, camera** cam, hittable** world, hittable** lights, curandState* rand) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= w || j >= h) {
        return;
    }

    // pixel color info
    int pixel = j * w + i;
    curandState local_rand = rand[pixel];
    vec3 col(0, 0, 0);
    for (int s = 0; s < samples; s++) {
        float u = float(i + curand_uniform(&local_rand)) / float(w);
        float v = float(j + curand_uniform(&local_rand)) / float(h);
        ray r = (*cam)->get_ray(u, v, &local_rand);
        col += ray_color(r, world, lights, &local_rand);
    }
    fb[pixel] = col / samples;
}

__global__ void free_world(hittable** d_list, hittable** d_world, hittable** lights, camera** cam) {
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
    delete *lights;
    delete cam;
}

int main() {
    const int width = 1200;
    const int height = 800;
    const int channel_num = 3;
    const int num_pixels = width * height * channel_num;
    size_t fb_size = num_pixels * sizeof(vec3);

    // Setup world
    std::cerr << "Setting up world" << std::endl;
    hittable** obj_list;
    checkCudaErrors(cudaMalloc((void**)&obj_list, 2 * sizeof(hittable *)));
    hittable** world;
    checkCudaErrors(cudaMalloc((void**)&world, 2 * sizeof(hittable *)));
    camera** cam;
    checkCudaErrors(cudaMalloc((void **)&cam, sizeof(camera *)));
    hittable** lights;
    checkCudaErrors(cudaMalloc((void**)&lights, sizeof(hittable *)));

    create_world<<<1, 1>>>(obj_list, world, lights, cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Setup and render
    std::cerr << "Initializing render" << std::endl;
    curandState* rand_state;
    checkCudaErrors(cudaMalloc((void**)&rand_state, num_pixels * sizeof(curandState)));

    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    const int cr_x = 16;
    const int cr_y = 16;
    const int samples_per_pixel = 1000;
    
    clock_t start, stop;
    start = clock();

    dim3 blocks(width/cr_x + 1, height/cr_y + 1);
    dim3 threads(cr_x, cr_y);
    render_init<<<blocks, threads>>>(width, height, rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cerr << "Starting render" << std::endl;
    render<<<blocks, threads>>>(fb, width, height, samples_per_pixel, cam, world, lights, rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Finished render\n";
    std::cerr << "Took " << timer_seconds << " seconds" << std::endl;

    // Frame buffer --> image
    auto* pixels = new unsigned char[width * height * channel_num];
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            size_t idx = j * width + i;

            float r = fb[idx].x();
            float g = fb[idx].y();
            float b = fb[idx].z();

			r = std::sqrt(r);
			g = std::sqrt(g);
			b = std::sqrt(b);

            pixels[idx * channel_num] = (unsigned char)(255.99f * r);
            pixels[idx * channel_num + 1] = (unsigned char)(255.99f * g);
            pixels[idx * channel_num + 2] = (unsigned char)(255.99f * b);
        }
    }

    stbi_flip_vertically_on_write(true);
	const int err = stbi_write_jpg("test_image.jpg", width, height, channel_num, pixels, 100);
	if (err) {
		std::cerr << "Image saved successfully\n";
	} else {
		std::cerr << "ERROR::Write_JPG: Image failed to save with code " << err << '\n';
	}

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(obj_list, world, lights, cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(obj_list));
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(fb));
    delete[] pixels;

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();

    system("pause");
    return 0;
}