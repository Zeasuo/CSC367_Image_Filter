/* ------------
 * This code is provided solely for the personal and private use of
 * students taking the CSC367H5 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited.
 * All forms of distribution of this code, whether as given or with
 * any changes, are expressly prohibited.
 *
 * Authors: Bogdan Simion, Felipe de Azevedo Piovezan
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2019 Bogdan Simion
 * -------------
 */

#ifndef __KERNELS__H
#define __KERNELS__H


/* TODO: you may want to change the signature of some or all of those functions,
 * depending on your strategy to compute min/max elements.
 * Be careful: "min" and "max" are names of CUDA library functions
 * unfortunately, so don't use those for variable names.*/
 __device__ inline int32_t apply2dGPU(const int8_t *f, int32_t dimension, const int32_t *original, int32_t width, int32_t height, int row, int column){
    int32_t pixel = 0;
    int32_t upper_left_row = row - dimension/2;
    int32_t upper_left_column = column - dimension/2;
    for (int r = 0; r < dimension; r ++) { // for each row in the filter
        for (int c = 0; c < dimension; c ++) { // for each col in the filter
            int32_t curr_row = upper_left_row + r; 
            int32_t curr_col = upper_left_column + c;
            if (curr_row >= 0 && curr_col >= 0 && curr_row < height && curr_col < width) {
                int index = curr_row * width + curr_col; // coordinate of the current pixel
                pixel += original[index] * f[r * dimension + c];
            }
        }
    }
    return pixel;
}
bool my_calculate_blocks_and_threads(int n, int &blocks, int &threads);
template <unsigned int blockSize> __global__ void first_reduction(int32_t *indata, int32_t *smallest, int32_t *largest, int n);
template <unsigned int blockSize> __global__ void reduction(int32_t *smallest, int32_t *largest, int n);
void gpu_switch_threads(int pixelCount, int numThreads, int numBlocks, int32_t *indata, int32_t *min, int32_t *max, int first);

void run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height);

void run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel1(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize1(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

void run_kernel2(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel2(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize2(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

void run_kernel3(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel3(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize3(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

void run_kernel4(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel4(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize4(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

void run_kernel5(int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
/* This is your own kernel, you should decide which parameters to add
   here*/
__global__ void kernel5(int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

#endif
