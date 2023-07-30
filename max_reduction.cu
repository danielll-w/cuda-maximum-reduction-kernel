#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
    block_max_reduction_kernel 

    Kernel to find the max across all threads in a block 

    Inputs: matrix of values and matrix to hold max value(s) for each block
    *** MUST BE PADDED TO MULTIPLE OF 32 IN DIMENSION BEING REDUCED
    *** CURRENTLY ONLY FINDS MAX OF POSITIVE NUMBERS
    
    Outputs: None...new matrix with one less dimension filled with the max of each block 

*/
__global__ void block_max_reduction_kernel(double* grid, double* row_max_grid) {

    // shared memory so we can compare the max across warps
    // residual matrix was padded to have dimensions in multiples of 32 so
    // length of array is number of threads / 32 i.e. we can store a max for each
    // up to 32 warps (since max threads is 1024)
    extern __shared__ double warp_max[];

    // residual at grid point and variable to hold value to compare to 
    double x = grid[blockIdx.x * blockDim.x + threadIdx.x];
    double x_comp;
    
    // do comparisons in chunks of 32
    // each set of 32 points will have max across those 32 points
    int mask, level;
    for (mask = 1, level = 0; level < 6; mask *= 2, ++level) {
        x_comp = __shfl_xor_sync(0xFFFFFFFF, x, mask);
        x = x > x_comp ? x : x_comp;
    }

    // fill shared memory array with maxes of each 32 element warp
    if (threadIdx.x % 32 == 0) {
        warp_max[threadIdx.x / warpSize] = x;
    }

    __syncthreads();

    // get max across warps in a given block 
    // first threads from 0 to (num threads / 32) - 1 get values stored in warp_max and
    // every other thread stores 0 
    x = (threadIdx.x < blockDim.x / warpSize) ? warp_max[threadIdx.x] : 0;

    // find the max for this block 
    for (mask = 1, level = 0; level < 6; mask *= 2, ++level) {
        x_comp = __shfl_xor_sync(0xFFFFFFFF, x, mask);
        x = x > x_comp ? x : x_comp;
    }

    // save max of block to output array
    if (threadIdx.x == 0) {
        row_max_grid[blockIdx.x] = x;
    } 
}

int main(int argc, char* argv[]) {

    // device handle and properties
    int dev;
    cudaDeviceProp prop;

    // set all entries to 0 to mean no preference
    memset(&prop, 0, sizeof(cudaDeviceProp));

    // look for GPU with at least 13 cores
    prop.multiProcessorCount = 13;

    // choose and set GPU
    cudaChooseDevice(&dev, &prop);
    cudaSetDevice(dev);

    // x and y dimensions
    int N_x = 34;
    int N_y = 34; 

    // pad to get multiples of 32
    N_x = N_x % 32 == 0 ? N_x : N_x + 32 - (N_x % 32); 
    N_y = N_y % 32 == 0 ? N_y : N_y + 32 - (N_y % 32); 

    // create grid grid to take max over 
    double* grid = (double *) malloc(N_x * N_y * sizeof(double));

    for (int i = 0; i < N_y; ++i) {
        for (int j = 0; j < N_x; ++j) {
            grid[i * N_x + j]  = i * N_x + j;        
        }
    }

    // copy array to device
    double* dev_grid;
    cudaMalloc((void**)&dev_grid, N_x * N_y * sizeof(double));
    cudaMemcpy(dev_grid, grid, N_x * N_y * sizeof(double), cudaMemcpyHostToDevice);

    // allocate memory for grid at each point and for max grid for each row on device
    double* dev_row_max_grid; 
    cudaMalloc((void**)&dev_row_max_grid, N_y * sizeof(double));

    // max grid on host and device 
    double max_grid; 
    double* dev_max_grid;
    cudaMalloc((void**)&dev_max_grid, sizeof(double));

    // call kernel
    block_reduction<<<N_y, N_x, 32 * sizeof(double)>>>(dev_grid, dev_row_max_grid);
    block_reduction<<<1, N_y, 32 * sizeof(double)>>>(dev_row_max_grid, dev_max_grid);

    // copy array of row maxes and overall max back to host
    double* row_max_grid = (double *) malloc(N_y * sizeof(double));
    cudaMemcpy(row_max_grid, dev_row_max_grid, N_y * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_grid, dev_max_grid, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Max: %f\n", max_grid);

    // free memory on device 
    cudaFree(dev_grid);
    cudaFree(dev_max_grid);
    cudaFree(dev_row_max_grid);

    // free memory on host 
    free(grid);

    return 0;

}
