#pragma once

class THClState;
class CLKernel;
class THClTensor;
class CLWrapper;

// this uses ints for all the long/int type things
// we can create a new version later that uses longs
// for now, you need to make sure:
// - anything inty in the kernel parameters is an int, not a long etc
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
class THClKernels {
  THClState *state;
  CLKernel *kernel;

public:
  THClKernels(THClState *state, CLKernel *kernel);

  THClKernels *in(THClTensor *tensor);
  THClKernels *inout(THClTensor *tensor);
  THClKernels *out(THClTensor *tensor);

  THClKernels *in(CLWrapper *wrapper);
  THClKernels *inout(CLWrapper *wrapper);
  THClKernels *out(CLWrapper *wrapper);

  THClKernels *in(int value);
  THClKernels *in(float value);
};

