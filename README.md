# Maximum Reduction Kernel

## Introduction

This CUDA code contains a kernel to find the maximum of an array. It can be used to find the maximum of a multidimensional
array by being called in succession. In essence, what is happening is that it finds the max across blocks and saves the
results in an array of length equal to the number of blocks the kernel was invoked with. So, for a NxM 2D grid, you could
call it first with N blocks and M threads to get the max in each of the N rows. Then call it with 1 block and N threads
to get the max for the whole array.

## Usage

The code has the kernel at the top and a main function showing an example of how it can be used. Two notes. First, the
kernel is not yet equipped to handle negative numbers. It originally was written to find the maximum residual across
grid points in a discretized PDE being solved using SOR. Second, whichever dimension you reduce should be an array that
is padded out with zeros so that the dimension is a multiple of 32.

## References

The idea to work with finding the block maxima comes from this blog post that discusses reduction kernels for summing.

https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/

For more on warp shuffles, and where I learned about using the xor shuffle, see *Introduction to High Performance Computing*
by David Chopp.

## Author
- Dan Weiss (2023)