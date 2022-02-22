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

#include <cuda.h>
#include "kernels.h"
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <stdint.h>
#define max_threads_block 1024
#define min(x ,y) ((x<y)?x:y)
#define max(x ,y) ((x>y)?x:y)

__constant__ int8_t log_filter[] = {
    0, 1, 1, 2, 2, 2,   1,   1,   0, 1, 2, 4, 5, 5,   5,   4,   2,
    1, 1, 4, 5, 3, 0,   3,   5,   4, 1, 2, 5, 3, -12, -24, -12, 3,
    5, 2, 2, 5, 0, -24, -40, -24, 0, 5, 2, 2, 5, 3,   -12, -24, -12,
    3, 5, 2, 1, 4, 5,   3,   0,   3, 5, 4, 1, 1, 2,   4,   5,   5,
    5, 4, 2, 1, 0, 1,   1,   2,   2, 2, 1, 1, 0,
};

void run_kernel5(int32_t dimension, const int32_t *input, int32_t *output, int32_t width, int32_t height) {
    // Figure out how to split the work into threads and call the kernel below.
    int nPixel = width * height;
    int32_t threads_needed = min(max_threads_block, nPixel);
    int32_t block_needed = (nPixel + threads_needed -1)/threads_needed;
    int32_t reduction_blocks;
    int32_t reduction_threads;
    int32_t iteration_n = nPixel;
    dim3 threads(threads_needed, 1);
    dim3 blocks(block_needed);
    kernel5 <<<blocks, threads>>> (dimension, input, output, width, height);
    
    int32_t *global_mins;
    int32_t *global_maxs;
    cudaMalloc(&global_mins, width*height*sizeof(int32_t));
    cudaMalloc(&global_maxs, width*height*sizeof(int32_t));
    bool should_repeat = my_calculate_blocks_and_threads(iteration_n, reduction_blocks, reduction_threads);
    gpu_switch_threads(iteration_n, reduction_threads, reduction_blocks, output, global_mins, global_maxs, 1);
    while(should_repeat){
        iteration_n = reduction_blocks;
        should_repeat = my_calculate_blocks_and_threads(iteration_n, reduction_blocks, reduction_threads);
        gpu_switch_threads(iteration_n, reduction_threads, reduction_blocks, output, global_mins, global_maxs, 0);//the output parameter passed here does not matter as it will not be used in the function
    }

    normalize5 <<<blocks, threads>>> (output, width, height, global_mins, global_maxs);
    cudaFree(global_mins);
    cudaFree(global_maxs);
}

__global__ void kernel5(int32_t dimension, const int32_t *input, int32_t *output, int32_t width, int32_t height) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int32_t row = idx / width;
    int32_t col = idx % width;
    int index = (row*width)+col;
    if (index < width*height && idx < width*height){
        output[index] = apply2dGPU(log_filter, dimension, input, width, height, row, col);
    }
}

__global__ void normalize5(int32_t *image, int32_t width, int32_t height, int32_t *smallest, int32_t *biggest) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int32_t row = idx / width;
    int32_t col = idx % width;
    int index = row*width+col;
    if (index < height * width){
        if (smallest[0] == biggest[0]) {
            return;
        }
        image[index] = ((image[index] - smallest[0]) * 255) / (biggest[0] - smallest[0]);
    }
}
