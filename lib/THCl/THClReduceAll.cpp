#include "THClTensorInfoCl.h"

#include <iostream>
using namespace std;

// Reduces the entire tensor to one floating-point value. `out` points
// to host-resident memory.
bool THClTensor_reduceAll(THClState* state,
                            THClTensor* in,
                            const ModifyOp& modifyOp,
                            const ReduceOp& reduceOp,
                            float init,
                            float* out,
                            int outOnDevice) {
  long inElements = THClTensor_nElement(state, in);

  if (THClTensor_nDimension(state, in) > MAX_CLTORCH_DIMS) {
    return false;
  }

  if (THClTensor_nDimension(state, in) == 0) {
    // Zero-dim tensor; do nothing
    *out = init;
    return true;
  }

  float* devOut = out;
  if (!outOnDevice) {
    // Use the stream-specific scratch space for the reduction kernel
    // to write out its value
    devOut = (float*) THClState_getCurrentDeviceScratchSpace(state);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, IN)                                           \
  callReduceAll<ModifyOp, ReduceOp, TYPE, IN>(                          \
    state, inInfo, inElements, init, modifyOp, reduceOp, devOut);

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
  if (!outOnDevice) {
    cudaMemcpy(out, devOut, sizeof(float), cudaMemcpyDeviceToHost);
  }

  return true;
}

std::string THClReduceAll_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClReduceAll.cl" )
  // ]]]
  // generated using cog, from THClReduceAll.cl:
  const char * kernelSource =  
  "IndexType getStartIndex(IndexType totalSize) {\n" 
  "  IndexType sizePerBlock = THCCeilDiv(totalSize, (IndexType) gridDim.x);\n" 
  "  return blockIdx.x * sizePerBlock;\n" 
  "}\n" 
  "\n" 
  "IndexType getEndIndex(IndexType totalSize) {\n" 
  "  IndexType sizePerBlock = THCCeilDiv(totalSize, (IndexType) gridDim.x);\n" 
  "  return min((IndexType) ((blockIdx.x + 1) * sizePerBlock), totalSize);\n" 
  "}\n" 
  "\n" 
  "// Kernel that handles an entire reduction of a tensor in one pass\n" 
  "kernel void\n" 
  "THClTensor_reduceAll(global TensorInfoCl *in,\n" 
  "                     global float *in_data,\n" 
  "                     IndexType totalElements,\n" 
  "                     float init,\n" 
  "                     global float* out,\n" 
  "                     local float *smem) {\n" 
  "  // With a block-wide stride, have each thread perform its own reduction.\n" 
  "  float r = init;\n" 
  "  for (IndexType i = threadIdx.x; i < totalElements; i += blockDim.x) {\n" 
  "    const IndexType inOffset = IndexToOffset<IndexType, ADims>::get(i, in);\n" 
  "    r = reduceOp(r, modifyOp(in.data[inOffset]));\n" 
  "  }\n" 
  "\n" 
  "  // Reduce within the block\n" 
  "//  extern __shared__ float smem[];\n" 
  "  r = reduceBlock<float, ReduceOp>(smem, blockDim.x, r, reduceOp, init);\n" 
  "\n" 
  "  if (threadIdx.x == 0) {\n" 
  "    // Write out reduced value\n" 
  "    out[0] = r;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "// Kernel that handles an entire reduction of a tensor in two passes\n" 
  "kernel void\n" 
  "THClTensor_reduceAllPass1(global TensorInfoCl *in,\n" 
  "                          global float *in_data,\n" 
  "                          IndexType totalElements,\n" 
  "                          float init,\n" 
  "                          global float* scratchSpace,\n" 
  "                          local float *smem) {\n" 
  "  const IndexType startIndex = getStartIndex<IndexType>(totalElements);\n" 
  "  const IndexType endIndex = getEndIndex<IndexType>(totalElements);\n" 
  "\n" 
  "  // With a block-wide stride, have each thread perform its own reduction.\n" 
  "  float r = init;\n" 
  "  for (IndexType i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {\n" 
  "    const IndexType inOffset = IndexToOffset<IndexType, ADims>::get(i, in);\n" 
  "    r = reduceOp(r, modifyOp(in.data[inOffset]));\n" 
  "  }\n" 
  "\n" 
  "  // Reduce within the block\n" 
  "//  extern __shared__ float smem[];\n" 
  "  r = reduceBlock<float, ReduceOp>(smem, blockDim.x, r, reduceOp, init);\n" 
  "\n" 
  "  if (threadIdx.x == 0) {\n" 
  "    // Write out block-wide reduced value\n" 
  "    scratchSpace[blockIdx.x] = r;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "template <typename ReduceOp, typename IndexType>\n" 
  "__global__ void\n" 
  "THClTensor_reduceAllPass2(int numPass1Blocks,\n" 
  "                            float init,\n" 
  "                            ReduceOp reduceOp,\n" 
  "                            float* scratchSpace,\n" 
  "                            float* out) {\n" 
  "  float r = init;\n" 
  "  if (threadIdx.x < numPass1Blocks) {\n" 
  "    r = scratchSpace[threadIdx.x];\n" 
  "  }\n" 
  "\n" 
  "  // Reduce within the block\n" 
  "  extern __shared__ float smem[];\n" 
  "  r = reduceBlock<float, ReduceOp>(smem, numPass1Blocks, r, reduceOp, init);\n" 
  "\n" 
  "  if (threadIdx.x == 0) {\n" 
  "    *out = r;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

