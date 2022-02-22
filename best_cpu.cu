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

#include "kernels.h"

#include <pthread.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <sched.h>
#define numThread 8

typedef struct filter_t {
    int32_t dimension;
    const int8_t *matrix;
} filter;

typedef struct common_work_t
{
    const filter *f;
    const int32_t *original_image;
    int32_t *output_image;
    int32_t width;
    int32_t height;
    int32_t max_threads;
    pthread_barrier_t barrier;
} common_work;

typedef struct work_t
{
    common_work *common;
    int32_t id;
} work;
pthread_mutex_t mutex;

int32_t global_min;
int32_t global_max;
/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest,
                     int32_t largest) {
  if (smallest == largest) {
    return;
  }

  target[pixel_idx] =
      ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}


int32_t apply2d(const filter *f, const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int row, int column)
{
    int32_t pixel = 0;
    int32_t upper_left_row = row - f->dimension/2;
    int32_t upper_left_column = column - f->dimension/2;
    for (int r = 0; r < f->dimension; r ++) { 
        for (int c = 0; c < f->dimension; c ++) { 
            int32_t curr_row = upper_left_row + r; 
            int32_t curr_col = upper_left_column + c;
            if (curr_row >= 0 && curr_col >= 0 && curr_row < height && curr_col < width) {
                int index = curr_row * width + curr_col;
                pixel += original[index] * f->matrix[r * f->dimension + c];
            }
        }
    }
    return pixel;
}

void *sharding_work(void *param) {
  /* Your algorithm is essentially:
   *  1- Apply the filter on the image
   *  2- Wait for all threads to do the same
   *  3- Calculate global smallest/largest elements on the resulting image
   *  4- Scale back the pixels of the image. For the non work queue
   *      implementations, each thread should scale the same pixels
   *      that it worked on step 1.
   */
  work w = *(work*) param;
  int32_t row, col;
  int32_t height = w.common->height;
  int32_t width = w.common->width;
  int32_t max_threads = w.common->max_threads;
  const filter *f = w.common->f;
  const int32_t *original_image = w.common->original_image;
  int32_t *output_image = w.common->output_image;

  int32_t num_row;
  int32_t local_min = INT_MAX;
  int32_t local_max = INT_MIN;
  srandom(time(NULL));
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET((w.id+1) % get_nprocs(), &set);
	if (sched_setaffinity(getpid(), sizeof(set), &set) != 0) {
		perror("sched_setaffinity");
		return NULL;
	}
  if (height % max_threads == 0){
    num_row = height / max_threads;
  }
  else{
    num_row = (height + max_threads - 1)/max_threads;
  }
  //Return instantly if assigned row/column is already out of bounds
  if(w.id*num_row >= height){
    pthread_barrier_wait(&(w.common->barrier));
    return NULL;
  }
  for (row=w.id*num_row; row<(w.id+1)*num_row && row<height; row++){
    for(col = 0; col < width; col ++){
      output_image[row * width + col] = apply2d(f, original_image, output_image, width, height, row, col);
      if (output_image[row * width + col] > local_max){
        local_max = output_image[row * width + col];
      }
      if (output_image[row * width + col] < local_min){
        local_min = output_image[row * width + col];
      }
    }
  }
  //update global max and min
  pthread_mutex_lock(&mutex);
  if (local_max > global_max){
    global_max = local_max;
  }
  if (local_min < global_min){
    global_min = local_min;
  }
  pthread_mutex_unlock(&mutex);
  //wait for other threads to finish
  pthread_barrier_wait(&(w.common->barrier));
  for (row=w.id*num_row; row<(w.id+1)*num_row && row<height; row++){
    for(col = 0; col < width; col ++){
      normalize_pixel(output_image, row*width+col, global_min, global_max);
    }
  }

  return NULL;
}

void run_best_cpu(const int8_t *fil, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height) {
  int num_threads = 8;
  // initialize common work
  global_min = INT_MAX;
  global_max = INT_MIN;
  common_work* cw = (common_work*)malloc(sizeof(common_work));
  const filter f = {dimension, fil};
  cw->f = &f;
  cw->original_image = input;
  cw->output_image = output;
  cw->width = width;
  cw->height = height;
  cw->max_threads = num_threads;
  pthread_t threads[num_threads];

  pthread_barrier_init(&(cw->barrier) ,NULL, num_threads);

  work** threads_work = (work**)malloc(sizeof(work*) * num_threads);
  for (int i = 0; i < num_threads; i ++) {
    threads_work[i] = (work *) malloc(sizeof(work));
    threads_work[i]->common = cw;
    threads_work[i]->id = i;
  }

  for (int i = 0; i < num_threads; ++i) {
    pthread_create(&threads[i], NULL, sharding_work, (void *)threads_work[i]);
  }

  for (int i = 0; i < num_threads; ++i) {
    pthread_join(threads[i], NULL);
  }

  for (int i = 0; i < num_threads; ++i) {
    free(threads_work[i]);
  }

  free(threads_work);
  free(cw);

}
