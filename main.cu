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

#include <stdio.h>
#include <string>
#include <unistd.h>

#include "pgm.h"
#include "clock.h"
#include "kernels.h"


/* laplacian */
int8_t lp3_m[] =
    {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
    };

int8_t lp5_m[] =
    {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 24, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
    };

/* Laplacian of gaussian */
int8_t log_m[] =
    {
        0, 1, 1, 2, 2, 2, 1, 1, 0,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        2, 5, 0, -24, -40, -24, 0, 5, 2,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        0, 1, 1, 2, 2, 2, 1, 1, 0,
    };
/* Use this function to print the time of each of your kernels.
 * The parameter names are intuitive, but don't hesitate to ask
 * for clarifications.
 * DO NOT modify this function.*/
void print_run(float time_cpu, int kernel, float time_gpu_computation,
               float time_gpu_transfer_in, float time_gpu_transfer_out) {
  printf("%12.6f ", time_cpu);
  printf("%5d ", kernel);
  printf("%12.6f ", time_gpu_computation);
  printf("%14.6f ", time_gpu_transfer_in);
  printf("%15.6f ", time_gpu_transfer_out);
  printf("%13.2f ", time_cpu / time_gpu_computation);
  printf("%7.2f\n", time_cpu / (time_gpu_computation + time_gpu_transfer_in +
                                time_gpu_transfer_out));
}

int main(int argc, char **argv) {
  int c;
  std::string input_filename, cpu_output_filename, base_gpu_output_filename;
  if (argc < 3) {
    printf("Wrong usage. Expected -i <input_file> -o <output_file>\n");
    return 0;
  }

  while ((c = getopt(argc, argv, "i:o:")) != -1) {
    switch (c) {
    case 'i':
      input_filename = std::string(optarg);
      break;
    case 'o':
      cpu_output_filename = std::string(optarg);
      base_gpu_output_filename = std::string(optarg);
      break;
    default:
      return 0;
    }
  }

  pgm_image source_img;
  init_pgm_image(&source_img);

  if (load_pgm_from_file(input_filename.c_str(), &source_img) != NO_ERR) {
    printf("Error loading source image.\n");
    return 0;
  }

  /* Do not modify this printf */
  printf("CPU_time(ms) Kernel GPU_time(ms) TransferIn(ms) TransferOut(ms) "
         "Speedup_noTrf Speedup\n");

  /* TODO: run your CPU implementation here and get its time. Don't include
   * file IO in your measurement.*/
  /* For example: */
  float time_cpu;
  {
    std::string cpu_file = cpu_output_filename;
    pgm_image cpu_output_img;
    copy_pgm_image_size(&source_img, &cpu_output_img);
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    run_best_cpu(log_m, 9, source_img.matrix, cpu_output_img.matrix, source_img.width, source_img.height); 
    clock_gettime(CLOCK_MONOTONIC, &stop);
    time_cpu = ((stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1000000000) * 1000;
    //print_run(time_cpu, 0, 0, 0, 0); //This line is for generating graphs, uncomment it when generating graphs
    save_pgm_to_file(cpu_file.c_str(), &cpu_output_img);
    destroy_pgm_image(&cpu_output_img);
  }

  /* TODO:
   * run each of your gpu implementations here,
   * get their time,
   * and save the output image to a file.
   * Don't forget to add the number of the kernel
   * as a prefix to the output filename:
   * Print the execution times by calling print_run().
   */

  /* For example: */
  {
//------------------------Kernel 1--------------------------------------------------------------------------------------------
    // Start time
    // run_kernel1(args...);  // From kernels.h
    // End time
    // print_run(args...)     // Defined on the top of this file
    std::string gpu_file = "1" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *deviceIn,*deviceOut;
    int8_t *deviceFilter;
    int nPixel = gpu_output_img.width*gpu_output_img.height;
    cudaMalloc((void**)&deviceIn,nPixel*sizeof(int32_t));
    cudaMalloc((void**)&deviceOut,nPixel*sizeof(int32_t));
    cudaMalloc((void**)&deviceFilter,9*9*sizeof(int8_t));

    Clock clock;
    float transfer_in, transfer_out, execution_time;

    //Transfer in data time for kernel
    clock.start();
    cudaMemcpy(deviceIn,source_img.matrix,nPixel*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOut,source_img.matrix,nPixel*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter,log_m,9*9*sizeof(int8_t),cudaMemcpyHostToDevice);
    transfer_in = clock.stop();

    //Execution time for kernel
    clock.start();
    run_kernel1(deviceFilter,9,deviceIn,deviceOut,gpu_output_img.width,gpu_output_img.height);
    execution_time = clock.stop();

    //Transfer out data time for kernel
    clock.start();
    cudaMemcpy(gpu_output_img.matrix,deviceOut,nPixel*sizeof(int32_t), cudaMemcpyDeviceToHost);
    transfer_out = clock.stop();

    //Clean Up
    cudaFree(deviceIn);
    cudaFree(deviceOut);
    cudaFree(deviceFilter);

    print_run(time_cpu, 1, execution_time, transfer_in, transfer_out);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
  }

  /* Repeat that for all 5 kernels. Don't hesitate to ask if you don't
   * understand the idea. */
  {
    //------------------------Kernel 2--------------------------------------------------------------------------------------------
    // Start time
    // run_kernel1(args...);  // From kernels.h
    // End time
    // print_run(args...)     // Defined on the top of this file
    std::string gpu_file = "2" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *deviceIn,*deviceOut;
    int8_t *deviceFilter;
    int nPixel = gpu_output_img.width*gpu_output_img.height;
    cudaMalloc((void**)&deviceIn,nPixel*sizeof(int32_t));
    cudaMalloc((void**)&deviceOut,nPixel*sizeof(int32_t));
    cudaMalloc((void**)&deviceFilter,9*9*sizeof(int8_t));

    Clock clock;
    float transfer_in, transfer_out, execution_time;

    //Transfer in data time for kernel
    clock.start();
    cudaMemcpy(deviceIn,source_img.matrix,nPixel*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOut,source_img.matrix,nPixel*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter,log_m,9*9*sizeof(int8_t),cudaMemcpyHostToDevice);
    transfer_in = clock.stop();

    //Execution time for kernel
    clock.start();
    run_kernel2(deviceFilter,9,deviceIn,deviceOut,gpu_output_img.width,gpu_output_img.height);
    execution_time = clock.stop();

    //Transfer out data time for kernel
    clock.start();
    cudaMemcpy(gpu_output_img.matrix,deviceOut,nPixel*sizeof(int32_t), cudaMemcpyDeviceToHost);
    transfer_out = clock.stop();

    //Clean Up
    cudaFree(deviceIn);
    cudaFree(deviceOut);
    cudaFree(deviceFilter);

    print_run(time_cpu, 2, execution_time, transfer_in, transfer_out);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
  }
  {
//------------------------Kernel 3--------------------------------------------------------------------------------------------
    // Start time
    // run_kernel1(args...);  // From kernels.h
    // End time
    // print_run(args...)     // Defined on the top of this file
    std::string gpu_file = "3" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *deviceIn,*deviceOut;
    int8_t *deviceFilter;
    int nPixel = gpu_output_img.width*gpu_output_img.height;
    cudaMalloc((void**)&deviceIn,nPixel*sizeof(int32_t));
    cudaMalloc((void**)&deviceOut,nPixel*sizeof(int32_t));
    cudaMalloc((void**)&deviceFilter,9*9*sizeof(int8_t));

    Clock clock;
    float transfer_in, transfer_out, execution_time;

    //Transfer in data time for kernel
    clock.start();
    cudaMemcpy(deviceIn,source_img.matrix,nPixel*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOut,source_img.matrix,nPixel*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter,log_m,9*9*sizeof(int8_t),cudaMemcpyHostToDevice);
    transfer_in = clock.stop();

    //Execution time for kernel
    clock.start();
    run_kernel3(deviceFilter,9,deviceIn,deviceOut,gpu_output_img.width,gpu_output_img.height);
    execution_time = clock.stop();

    //Transfer out data time for kernel
    clock.start();
    cudaMemcpy(gpu_output_img.matrix,deviceOut,nPixel*sizeof(int32_t), cudaMemcpyDeviceToHost);
    transfer_out = clock.stop();

    //Clean Up
    cudaFree(deviceIn);
    cudaFree(deviceOut);
    cudaFree(deviceFilter);

    print_run(time_cpu, 3, execution_time, transfer_in, transfer_out);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
  }
  {
    //------------------------Kernel 4--------------------------------------------------------------------------------------------
    // Start time
    // run_kernel1(args...);  // From kernels.h
    // End time
    // print_run(args...)     // Defined on the top of this file
    std::string gpu_file = "4" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *deviceIn,*deviceOut;
    int8_t *deviceFilter;
    int nPixel = gpu_output_img.width*gpu_output_img.height;
    cudaMalloc((void**)&deviceIn,nPixel*sizeof(int32_t));
    cudaMalloc((void**)&deviceOut,nPixel*sizeof(int32_t));
    cudaMalloc((void**)&deviceFilter,9*9*sizeof(int8_t));

    Clock clock;
    float transfer_in, transfer_out, execution_time;

    //Transfer in data time for kernel
    clock.start();
    cudaMemcpy(deviceIn,source_img.matrix,nPixel*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOut,source_img.matrix,nPixel*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter,log_m,9*9*sizeof(int8_t),cudaMemcpyHostToDevice);
    transfer_in = clock.stop();

    //Execution time for kernel
    clock.start();
    run_kernel4(deviceFilter,9,deviceIn,deviceOut,gpu_output_img.width,gpu_output_img.height);
    execution_time = clock.stop();

    //Transfer out data time for kernel
    clock.start();
    cudaMemcpy(gpu_output_img.matrix,deviceOut,nPixel*sizeof(int32_t), cudaMemcpyDeviceToHost);
    transfer_out = clock.stop();

    //Clean Up
    cudaFree(deviceIn);
    cudaFree(deviceOut);
    cudaFree(deviceFilter);

    print_run(time_cpu, 4, execution_time, transfer_in, transfer_out);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
  }
  {
    //------------------------Kernel 5--------------------------------------------------------------------------------------------
    // Start time
    // run_kernel1(args...);  // From kernels.h
    // End time
    // print_run(args...)     // Defined on the top of this file
    std::string gpu_file = "5" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *deviceIn,*deviceOut;
    int8_t *deviceFilter;
    int nPixel = gpu_output_img.width*gpu_output_img.height;

    //Pinned Memory
    int32_t *new_source_img;
    cudaMallocHost((void**)&new_source_img, nPixel*sizeof(int32_t));
    memcpy(new_source_img, source_img.matrix, nPixel*sizeof(int32_t));

    cudaMalloc((void**)&deviceIn,nPixel*sizeof(int32_t));
    cudaMalloc((void**)&deviceOut,nPixel*sizeof(int32_t));
    cudaMalloc((void**)&deviceFilter,9*9*sizeof(int8_t));

    Clock clock;
    float transfer_in, transfer_out, execution_time;

    //Transfer in data time for kernel
    clock.start();
    cudaMemcpy(deviceIn,new_source_img,nPixel*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOut,new_source_img,nPixel*sizeof(int32_t), cudaMemcpyHostToDevice);
    transfer_in = clock.stop();

    //Execution time for kernel
    clock.start();
    run_kernel5(9,deviceIn,deviceOut,gpu_output_img.width,gpu_output_img.height);
    execution_time = clock.stop();

    //Transfer out data time for kernel
    clock.start();
    cudaMemcpy(gpu_output_img.matrix,deviceOut,nPixel*sizeof(int32_t), cudaMemcpyDeviceToHost);
    transfer_out = clock.stop();

    //Clean Up
    cudaFree(new_source_img);
    cudaFree(deviceIn);
    cudaFree(deviceOut);
    cudaFree(deviceFilter);

    print_run(time_cpu, 5, execution_time, transfer_in, transfer_out);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
  }
}
