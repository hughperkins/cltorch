#include "EasyCL.h"
#include "THClReduceAll.h"
#include "THClTypeParseTraits.h"

#include <iostream>
using namespace std;

// Reduces the entire tensor to one floating-point value. `out` points
// to host-resident memory.
bool THClTensor_reduceAll(THClState* state,
                            THClTensor* in,
                            const HasOperator2 *modifyOp,
                            const HasOperator3 *reduceOp,
                            float init,
                            float *p_result) {
  long inElements = THClTensor_nElement(state, in);

  if (THClTensor_nDimension(state, in) > MAX_CLTORCH_DIMS) {
    return false;
  }

  if (THClTensor_nDimension(state, in) == 0) {
    // Zero-dim tensor; do nothing
    *p_result = init;
    return true;
  }

//  CLWrapper* devOut = out;
//  float *devOut 
//  if (!outOnDevice) {
    // Use the stream-specific scratch space for the reduction kernel
    // to write out its value
  THClScratchSpace *scratch = THClState_getCurrentDeviceScratchSpace(state);
  CLWrapper *devOut = scratch->wrapper;
//  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, IN)                                           \
  callReduceAll<TYPE>(                          \
    state, IN, inInfo, inElements, init, modifyOp, reduceOp, devOut);

#define HANDLE_IN_CASE(TYPE, IN)                    \
  {                                                 \
    if (inInfo.isContiguous()) {                    \
      HANDLE_CASE(TYPE, -2);                        \
    } else {                                        \
      switch (IN) {                                 \
        case 1:                                     \
          HANDLE_CASE(TYPE, 1);                     \
          break;                                    \
        case 2:                                     \
          HANDLE_CASE(TYPE, 2);                     \
          break;                                    \
        case 3:                                     \
          HANDLE_CASE(TYPE, 3);                     \
          break;                                    \
        default:                                    \
          HANDLE_CASE(TYPE, -1);                    \
          break;                                    \
      }                                             \
    }                                               \
  }

  if (THCL_canUse32BitIndexMath(state, in)) {
    TensorInfo<unsigned int> inInfo(state, in);

    HANDLE_IN_CASE(unsigned int, inInfo.dims);
  } else {
    TensorInfo<unsigned long long> inInfo(state, in);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (inInfo.isContiguous()) {
      HANDLE_IN_CASE(unsigned long long, -2);
    } else {
      HANDLE_IN_CASE(unsigned long long, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_IN_CASE

  // If our destination is not on the device, copy the value back to
  // the host (synchronous!)
//  if (!outOnDevice) {
//    cudaMemcpy(out, devOut, sizeof(float), cudaMemcpyDeviceToHost);
//  }

//  THError("Not implemented");

  scratch->wrapper->copyToHost();
  *p_result = scratch->data[0];

  return true;
}

std::string THClReduceAll_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClReduceAll.cl" )
  // ]]]
  // generated using cog, from THClReduceAll.cl:
  const char * kernelSource =  
  "{{include_THClDeviceUtils}}\n" 
  "\n" 
  "float modifyOp(float _in1) {\n" 
  "  float _out;\n" 
  "  float *in1 = &_in1;\n" 
  "  float *out = &_out;\n" 
  "  {{modify_operation}};\n" 
  "  return _out;\n" 
  "}\n" 
  "\n" 
  "float reduceOp(float _in1, float _in2) {\n" 
  "  // I guess the compiler can sort this stuff out :-P\n" 
  "  float _out;\n" 
  "  float *in1 = &_in1;\n" 
  "  float *in2 = &_in2;\n" 
  "  float *out = &_out;\n" 
  "  {{reduce_operation}};\n" 
  "  return _out;\n" 
  "}\n" 
  "\n" 
  "{{include_THClReduceApplyUtils}}\n" 
  "\n" 
  "// Kernel that handles an entire reduction of a tensor in one pass\n" 
  "kernel void\n" 
  "THClTensor_reduceAll(global TensorInfoCl *in_info,\n" 
  "                     global float *in_data,\n" 
  "                     {{IndexType}} totalElements,\n" 
  "                     float init,\n" 
  "                     global float* out,\n" 
  "                     local float *smem) {\n" 
  "  // With a block-wide stride, have each thread perform its own reduction.\n" 
  "  float r = init;\n" 
  "  for ({{IndexType}} i = /*threadIdx.x*/ get_local_id(0); i < totalElements; i += /*blockDim.x*/get_local_size(0)) {\n" 
  "    const {{IndexType}} inOffset = IndexToOffset_{{1000 + dim1}}_get(i, in_info[0]);\n" 
  "    r = reduceOp(r, modifyOp(in_data[inOffset]));\n" 
  "  }\n" 
  "\n" 
  "  // Reduce within the block\n" 
  "//  extern __shared__ float smem[];\n" 
  "  r = reduceBlock(smem, /*blockDim.x*/ get_local_size(0), r, init);\n" 
  "\n" 
  "  if (/*threadIdx.x*/ get_local_id(0) == 0) {\n" 
  "    // Write out reduced value\n" 
  "    out[0] = r;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "{{IndexType}} getStartIndex({{IndexType}} totalSize) {\n" 
  "  {{IndexType}} sizePerBlock = THClCeilDiv(totalSize, ({{IndexType}}) /*gridDim.x*/ get_num_groups(0));\n" 
  "  return /*blockIdx.x*/ get_group_id(0) * sizePerBlock;\n" 
  "}\n" 
  "\n" 
  "{{IndexType}} getEndIndex({{IndexType}} totalSize) {\n" 
  "  {{IndexType}} sizePerBlock = THClCeilDiv(totalSize, ({{IndexType}}) /*gridDim.x*/ get_num_groups(0));\n" 
  "  return min(({{IndexType}}) ((/*blockIdx.x*/ get_group_id(0) + 1) * sizePerBlock), totalSize);\n" 
  "}\n" 
  "\n" 
  "// Kernel that handles an entire reduction of a tensor in two passes\n" 
  "kernel void\n" 
  "THClTensor_reduceAllPass1(global TensorInfoCl *in_info,\n" 
  "                          global float *in_data,\n" 
  "                          {{IndexType}} totalElements,\n" 
  "                          float init,\n" 
  "                          global float* scratchSpace,\n" 
  "                          local float *smem) {\n" 
  "  const {{IndexType}} startIndex = getStartIndex(totalElements);\n" 
  "  const {{IndexType}} endIndex = getEndIndex(totalElements);\n" 
  "\n" 
  "  // With a block-wide stride, have each thread perform its own reduction.\n" 
  "  float r = init;\n" 
  "  for ({{IndexType}} i = startIndex + /*threadIdx.x*/ get_local_id(0); i < endIndex; i += /*blockDim.x*/ get_local_size(0)) {\n" 
  "    const {{IndexType}} inOffset = IndexToOffset_{{1000 + dim1}}_get(i, in_info[0]);\n" 
  "    r = reduceOp(r, modifyOp(in_data[inOffset]));\n" 
  "  }\n" 
  "\n" 
  "  // Reduce within the block\n" 
  "//  extern __shared__ float smem[];\n" 
  "  r = reduceBlock(smem, /*blockDim.x*/ get_local_size(0), r, init);\n" 
  "\n" 
  "  if (/*threadIdx.x*/ get_local_id(0) == 0) {\n" 
  "    // Write out block-wide reduced value\n" 
  "    scratchSpace[/*blockIdx.x*/ get_group_id(0)] = r;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "kernel void THClTensor_reduceAllPass2(int numPass1Blocks,\n" 
  "                            float init,\n" 
  "                            global float* scratchSpace,\n" 
  "                            global float* out,\n" 
  "                            local float *smem) {\n" 
  "  float r = init;\n" 
  "  if (/*threadIdx.x*/ get_local_id(0) < numPass1Blocks) {\n" 
  "    r = scratchSpace[/*threadIdx.x*/ get_local_id(0)];\n" 
  "  }\n" 
  "\n" 
  "  // Reduce within the block\n" 
  "//  extern __shared__ float smem[];\n" 
  "  r = reduceBlock(smem, numPass1Blocks, r, init);\n" 
  "\n" 
  "  if (/*threadIdx.x*/ get_local_id(0) == 0) {\n" 
  "    out[0] = r;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

