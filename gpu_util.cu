#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "kernels.h"
#define min(x ,y) ((x<y)?x:y)
#define max(x ,y) ((x>y)?x:y)

bool my_calculate_blocks_and_threads(int n, int &blocks, int &threads){
    threads = 1024;
    blocks = (n + threads -1)/threads;
    if (n <= 512) threads = 512;
    if (n <= 256) threads = 256;
    if (n <= 128) threads = 128;
    if (n <= 64) threads = 64;
    if (n <= 32) threads = 32;
    if (n <= 16) threads = 16;
    if (n <= 8) threads = 8;
    if (n <= 4) threads = 4;
    if (n <= 2) threads = 2;
    
    return blocks != 1;
}

template <unsigned int blockSize>
__global__ void first_reduction(int32_t *indata, int32_t *smallest, int32_t *biggest, int n) {
    //sdata_min_max[0] = mins
    //sdata_min_max[1] = maxs
    __shared__ int32_t sdata_min_max[2][1024];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < n){
        sdata_min_max[0][tid] = indata[i];
        sdata_min_max[1][tid] = indata[i];
    }
    else{
        sdata_min_max[0][tid] = (1<<30);
        sdata_min_max[1][tid] = (1<<31);
    }

	__syncthreads();

	// do reduction in shared memory
	if (blockSize >= 1024) { 
        if (tid < 512){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+512]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+512]);
        } 
        __syncthreads();
    }
    if (blockSize >= 512) { 
        if (tid < 256){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+256]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+256]);
        } 
        __syncthreads();
    }
	if (blockSize >= 256) { 
        if (tid < 128){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+128]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+128]);
        } 
        __syncthreads();
    }
	if (blockSize >= 128) {	
        if (tid <  64){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+64]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+64]);
        } 
        __syncthreads();
    }
	if (blockSize >= 64) { 
        if (tid < 32){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+32]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+32]);
        }
        __syncthreads();
    }
	if (blockSize >= 32) { 
        if (tid < 16){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+16]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+16]);
        } 
        __syncthreads();
    }
	if (blockSize >= 16) {	
        if (tid < 8){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+8]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+8]);
        } 
        __syncthreads();
    }
   if (blockSize >= 8) { 
        if (tid < 4){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+4]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+4]);
        } 
        __syncthreads();
    }
	if (blockSize >= 4) { 
        if (tid < 2){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+2]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+2]);
        } 
        __syncthreads();
    }
	if (blockSize >= 2) {	
        if (tid < 1){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+1]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+1]);
        } 
        __syncthreads();
    }

	// write result for this block back to global memory
	if (tid == 0) {
        smallest[blockIdx.x] = sdata_min_max[0][0];
        biggest[blockIdx.x] = sdata_min_max[1][0];
    }
}
template <unsigned int blockSize>
__global__ void reduction(int32_t *smallest, int32_t *biggest, int n) {
    __shared__ int32_t sdata_min_max[2][1024];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n){
        sdata_min_max[0][tid] = smallest[i];
        sdata_min_max[1][tid] = biggest[i];
    }
    else{
        sdata_min_max[0][tid] = (1<<30);
        sdata_min_max[1][tid] = (1<<31);
    }
	__syncthreads();

	// do reduction in shared memory
	if (blockSize >= 1024) { 
        if (tid < 512){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+512]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+512]);
        } 
        __syncthreads();
    }
    if (blockSize >= 512) { 
        if (tid < 256){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+256]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+256]);
        } 
        __syncthreads();
    }
	if (blockSize >= 256) { 
        if (tid < 128){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+128]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+128]);
        } 
        __syncthreads();
    }
	if (blockSize >= 128) {	
        if (tid <  64){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+64]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+64]);
        } 
        __syncthreads();
    }
	if (blockSize >= 64) { 
        if (tid < 32){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+32]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+32]);
        }
        __syncthreads();
    }
	if (blockSize >= 32) { 
        if (tid < 16){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+16]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+16]);
        } 
        __syncthreads();
    }
	if (blockSize >= 16) {	
        if (tid < 8){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+8]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+8]);
        } 
        __syncthreads();
    }
   if (blockSize >= 8) { 
        if (tid < 4){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+4]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+4]);
        } 
        __syncthreads();
    }
	if (blockSize >= 4) { 
        if (tid < 2){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+2]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+2]);
        } 
        __syncthreads();
    }
	if (blockSize >= 2) {	
        if (tid < 1){ 
            sdata_min_max[0][tid] = min(sdata_min_max[0][tid], sdata_min_max[0][tid+1]);
            sdata_min_max[1][tid] = max(sdata_min_max[1][tid], sdata_min_max[1][tid+1]);
        } 
        __syncthreads();
    }

	// write result for this block back to global memory
	if (tid == 0) {
        smallest[blockIdx.x] = sdata_min_max[0][0];
        biggest[blockIdx.x] = sdata_min_max[1][0];
    }
}
void gpu_switch_threads(int nPixel, int numThreads, int numBlocks, int32_t *indata, int32_t *min, int32_t *max, int first){
  int shMemSize = 2 * 1024 * sizeof(int32_t);
  switch (numThreads)
  {
    case 1024:
      if (first == 1) {first_reduction<1024><<<numBlocks,numThreads,shMemSize>>>(indata, min, max,nPixel);}
      else {reduction<1024><<<numBlocks,numThreads,shMemSize>>>(min, max,nPixel);}
      break;
    case 512:
      if (first) {first_reduction<512><<<numBlocks,numThreads,shMemSize>>>(indata, min, max,nPixel);}
      else {reduction<512><<<numBlocks,numThreads,shMemSize>>>(min, max,nPixel);}
      break;
    case 256:
      if (first) {first_reduction<256><<<numBlocks,numThreads,shMemSize>>>(indata, min, max,nPixel);}
      else {reduction<256><<<numBlocks,numThreads,shMemSize>>>(min, max,nPixel);}
      break;
    case 128:
      if (first) {first_reduction<128><<<numBlocks,numThreads,shMemSize>>>(indata, min, max,nPixel);}
      else {reduction<128><<<numBlocks,numThreads,shMemSize>>>(min, max,nPixel);}
      break;
    case 64:
      if (first) {first_reduction<64><<<numBlocks,numThreads,shMemSize>>>(indata, min, max,nPixel);}
      else {reduction<64><<<numBlocks,numThreads,shMemSize>>>(min, max,nPixel);}
      break;
    case 32:
     if (first) {first_reduction<32><<<numBlocks,numThreads,shMemSize>>>(indata, min, max,nPixel);}
      else {reduction<32><<<numBlocks,numThreads,shMemSize>>>(min, max,nPixel);}
      break;
    case 16:
      if (first) {first_reduction<16><<<numBlocks,numThreads,shMemSize>>>(indata, min, max,nPixel);}
      else {reduction<16><<<numBlocks,numThreads,shMemSize>>>(min, max,nPixel);}
      break;
    case  8:
      if (first) {first_reduction<8><<<numBlocks,numThreads,shMemSize>>>(indata, min, max,nPixel);}
      else {reduction<8><<<numBlocks,numThreads,shMemSize>>>(min, max,nPixel);}
      break;
    case  4:
      if (first) {first_reduction<4><<<numBlocks,numThreads,shMemSize>>>(indata, min, max,nPixel);}
      else {reduction<4><<<numBlocks,numThreads,shMemSize>>>(min, max,nPixel);}
      break;
    case  2:
      if (first) {first_reduction<2><<<numBlocks,numThreads,shMemSize>>>(indata, min, max,nPixel);}
      else {reduction<2><<<numBlocks,numThreads,shMemSize>>>(min, max,nPixel);}
      break;
    case  1:
      if (first) {first_reduction<1><<<numBlocks,numThreads,shMemSize>>>(indata, min, max,nPixel);}
      else {reduction<1><<<numBlocks,numThreads,shMemSize>>>(min, max,nPixel);}
      break;
    default:
      exit(1);
  }
}