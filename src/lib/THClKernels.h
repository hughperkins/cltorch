#pragma once

#include <vector>

class THClState;
class CLKernel;
class THClTensor;
class CLWrapper;

#include "THClGeneral.h"
#include "THClReduceApplyUtils.h"

// inty types
// ==========
//
// this uses ints for all the long/int type things
// we can create a new version later that uses longs
// for now, you need to make surethat anything inty in the kernel parameters is an int, not a long etc
//
// Passing THClTensors
// ===================
// - when you are passing in a tensor, you need two parameters for each tensor,
//   in the kernel, eg lets say in the cuda, there is a kernel parameter
//
//     __global__ foo( float *src, ...
//
// This will become, in our kernel:
//
//     kernel foo( global float *src_data, int src_offset, ...
//
// thats it :-)  now just use an object of this class to pass in the data
// oh...in the kernel, when you use the src_data object, make sure to 
// add the offset.  Like:
//
//    src[i]
//
// ... in the cuda becomes:
//
//    src_data[src_offset + i]
//
//  ....in the opencl
//
// Passing THClTEnsorInfos
// =======================
//
// On the receiving side, there needs to be two global parameters. ie, if in cuda kernel 
// we have:
//
//   __global__ foo(THClTensorInfo<IndexType> mytensor, ...)
//
// on OpenCL kernel, we will have:
//
//  kernel foo(global THClTensorInfoCl *mytensor_info, global float *mytensor_data, ...)
//
// You'll also need to define THClTensorInfoCl struct in your kernel, eg by including
// code from include_THClReduceApplyUtils.cl, see THClApply.h for an example
//
// in, inout, out
// ==============
// Note on difference between 'in', 'out', 'inout':
// - 'inout' and 'out' will mark the CLWrapper gpu buffer as 'dirty',
//   needing to be
//   copied to host, if we want to work on host-side
// - 'out' will allocate the CLWrapper device-side buffer, if not already
//   allocated (in and inout will throw an error, if not allocated on device-side
//   already)
class THClKernels {
  THClState *state;
  CLKernel *kernel;

  std::vector< TensorInfoCl * >tensorInfoCls;

public:
  THClKernels(THClState *state, CLKernel *kernel);
  ~THClKernels();

  THClKernels *in(THClTensor *tensor);
  THClKernels *inout(THClTensor *tensor);
  THClKernels *out(THClTensor *tensor);

  template< typename IndexType >
  THClKernels *in(TensorInfo<IndexType>tensorInfo);
  template< typename IndexType >
  THClKernels *inout(TensorInfo<IndexType>tensorInfo);
  template< typename IndexType >
  THClKernels *out(TensorInfo<IndexType>tensorInfo);

  THClKernels *in(CLWrapper *wrapper);
  THClKernels *inout(CLWrapper *wrapper);
  THClKernels *out(CLWrapper *wrapper);

  THClKernels *in(int value);
  THClKernels *in(float value);

  THClKernels *localFloats(int count);

  void run(dim3 grid, dim3 block);  // uses cutorch-compatible dimensions
};

