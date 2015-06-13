#include <iostream>

#include "THClReduce.h"

using namespace std;

#define THCL_NONCONTIG_REDUCE_BLOCK_SIZE 32 * 16

#define REGISTER_PARSE_TYPE(X) template <> struct TypeParseTraits<X> \
    { static const char* name; } ; const char* TypeParseTraits<X>::name = #X


REGISTER_PARSE_TYPE(unsigned int);
REGISTER_PARSE_TYPE(unsigned long);

// since this is no longer a template, so move to .cpp

// Performs a reduction out[..., 0, ...] = reduce_i(modify(in[..., i, ...])) for
// all in where i and the out's 0 are indexed at dimension `dim`
bool THClTensor_reduceDim(THClState* state,
                            THClTensor* out,
                            THClTensor* in,
                            float init,
                            const HasOperator2 *modifyOp,
                            const HasOperator3 *reduceOp,
                            int dim) {
  long inElements = THClTensor_nElement(state, in);

  long reductionSize = THClTensor_size(state, in, dim);
  long reductionStride = THClTensor_stride(state, in, dim);
  long outElements = inElements / reductionSize;

  if (THClTensor_nDimension(state, out) > MAX_CLTORCH_DIMS ||
      THClTensor_nDimension(state, in) > MAX_CLTORCH_DIMS) {
    return false;
  }

  if (THClTensor_nDimension(state, in) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  // Is the reduction dimension contiguous? If so, then we can use a
  // shared memory reduction kernel to increase performance.
  bool contigReduction = (reductionStride == 1);

  dim3 block;
  dim3 grid;
  int smemSize = 0; // contiguous reduction uses smem
  if (contigReduction) {
    if (!getContigReduceGrid(outElements, grid)) {
      return false;
    }

    block = getContigReduceBlock(outElements, reductionSize);
    smemSize = sizeof(float) * block.x();
  } else {
    if (!getNoncontigReduceGrid(outElements, grid)) {
      return false;
    }

    block = getNoncontigReduceBlock();
  }

  // Resize out to correspond to the reduced size
  THLongStorage* sizes = THClTensor_newSizeOf(state, in);
  THLongStorage_set(sizes, dim, 1);
  THClTensor_resize(state, out, sizes, NULL);
  THLongStorage_free(sizes);

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, OUT, IN)                                      \
  if (contigReduction) {                                                \
    kernelLaunch_THClTensor_reduceContigDim<TYPE> (    \
        state, OUT, IN, outInfo, inInfo, (TYPE) reductionSize,                                 \
        (TYPE) outElements, init, modifyOp, reduceOp);                  \
    /* THClTensor_reduceContigDim<ModifyOp, ReduceOp, TYPE, OUT, IN>     \
      <<<grid, block, smemSize, THClState_getCurrentStream(state)>>>(    \
        outInfo, inInfo, (TYPE) reductionSize,                                 \
        (TYPE) outElements, init, modifyOp, reduceOp);  */                \
          THError("Not implemented");  \
  } else {                                                              \
     kernelLaunch_THClTensor_reduceNoncontigDim<TYPE> (  \
        state, OUT, IN, outInfo, inInfo, (TYPE) reductionStride, (TYPE) reductionSize,                \
        (TYPE) outElements, init, modifyOp, reduceOp);                   \
    /* THClTensor_reduceNoncontigDim<ModifyOp, ReduceOp, TYPE, OUT, IN>  \
      <<<grid, block, 0, THClState_getCurrentStream(state)>>>(           \
        outInfo, inInfo, (TYPE) reductionStride, (TYPE) reductionSize,                \
        (TYPE) outElements, init, modifyOp, reduceOp);  */                 \
      THError("Not implemented");   \
  }                                                                     \

#define HANDLE_IN_CASE(TYPE, OUT, IN)                   \
  {                                                     \
    if (inInfo.isContiguous()) {                        \
      HANDLE_CASE(TYPE, OUT, -2);                       \
    } else {                                            \
      switch (IN) {                                     \
        case 1:                                         \
          HANDLE_CASE(TYPE, OUT, 1);                    \
          break;                                        \
        case 2:                                         \
          HANDLE_CASE(TYPE, OUT, 2);                    \
          break;                                        \
        case 3:                                         \
          HANDLE_CASE(TYPE, OUT, 3);                    \
          break;                                        \
        default:                                        \
          HANDLE_CASE(TYPE, OUT, -1);                   \
          break;                                        \
      }                                                 \
    }                                                   \
  }

#define HANDLE_OUT_CASE(TYPE, OUT, IN)                \
  {                                                   \
    if (outInfo.isContiguous()) {                     \
      HANDLE_IN_CASE(TYPE, -2, IN);                   \
    } else {                                          \
      switch (OUT) {                                  \
        case 1:                                       \
          HANDLE_IN_CASE(TYPE, 1, IN);                \
          break;                                      \
        case 2:                                       \
          HANDLE_IN_CASE(TYPE, 2, IN);                \
          break;                                      \
        case 3:                                       \
          HANDLE_IN_CASE(TYPE, 3, IN);                \
          break;                                      \
        default:                                      \
          HANDLE_IN_CASE(TYPE, -1, IN);               \
          break;                                      \
      }                                               \
    }                                                 \
  }

  if (THCL_canUse32BitIndexMath(state, out) &&
      THCL_canUse32BitIndexMath(state, in)) {
    TensorInfo<unsigned int> outInfo(state, out);
    TensorInfo<unsigned int> inInfo(state, in, dim);

    HANDLE_OUT_CASE(unsigned int, outInfo.dims, inInfo.dims);
  } else {
    TensorInfo<unsigned long> outInfo(state, out);
    TensorInfo<unsigned long> inInfo(state, in, dim);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (outInfo.isContiguous() && inInfo.isContiguous()) {
      HANDLE_CASE(unsigned long, -2, -2);
    } else {
      HANDLE_CASE(unsigned long, -1, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  return true;
}

std::string getKernelSource() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClReduce.cl" )
  // ]]]
  // generated using cog, from THClReduce.cl:
  const char * kernelSource =  
  "// Threads per thread block\n" 
  "#define THCL_NONCONTIG_REDUCE_BLOCK_SIZE 32 * 16\n" 
  "\n" 
  "{{include_TensorInfoCl}}\n" 
  "\n" 
  "{{index_type}} getReduceNoncontigDimSliceIndex() {\n" 
  "  // Each thread handles one slice\n" 
  "  return getLinearBlockId<{{index_type}}>() * THCL_NONCONTIG_REDUCE_BLOCK_SIZE + threadIdx.x;\n" 
  "}\n" 
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
  "// Kernel that handles an entire reduction of a slice of a tensor per each thread\n" 
  "kernel void\n" 
  "THClTensor_reduceNoncontigDim(TensorInfoCl<{{index_type}}> out,\n" 
  "                              global float *out_data,\n" 
  "                              TensorInfoCl<{{index_type}}> in,\n" 
  "                              global float *in_data,\n" 
  "                              {{index_type}} reductionStride,\n" 
  "                              {{index_type}} reductionSize,\n" 
  "                              {{index_type}} totalSlices,\n" 
  "                              float init) {\n" 
  "  const {{index_type}} sliceIndex = getReduceNoncontigDimSliceIndex<{{index_type}}>();\n" 
  "\n" 
  "  if (sliceIndex >= totalSlices) {\n" 
  "    return;\n" 
  "  }\n" 
  "\n" 
  "  // Each thread picks a point in `out` and `in` for which it is\n" 
  "  // producing the reduction\n" 
  "  const {{index_type}} outOffset =\n" 
  "    IndexToOffset_{{1000 + dim1}}_get(sliceIndex, out);\n" 
  "  const {{index_type}} inBaseOffset =\n" 
  "    IndexToOffset_{{1000 + dim2}}_get(sliceIndex, in);\n" 
  "\n" 
  "  // For each point in reductionSize, reduce into `r`\n" 
  "  {{index_type}} inOffset = inBaseOffset;\n" 
  "\n" 
  "  float r = modifyOp(in_data[0]);\n" 
  "  inOffset += reductionStride;\n" 
  "\n" 
  "  for ({{index_type}} i = 1; i < reductionSize; ++i) {\n" 
  "    r = reduceOp(r, modifyOp(in_data[inOffset]));\n" 
  "    inOffsets += reductionStride;\n" 
  "  }\n" 
  "\n" 
  "  // Write out reduced value\n" 
  "  out.data[outOffset] = r;\n" 
  "}\n" 
  "\n" 
  "{{index_type}} getReduceContigDimSliceIndex() {\n" 
  "  // Each block handles one slice\n" 
  "  return getLinearBlockId<{{index_type}}>();\n" 
  "}\n" 
  "\n" 
  "// Kernel that handles an entire reduction of a slice of a tensor per\n" 
  "// each block\n" 
  "kernel void\n" 
  "THClTensor_reduceContigDim(TensorInfoCl<{{index_type}}> out,\n" 
  "                           global float *out_data,\n" 
  "                           TensorInfoCl<{{index_type}}> in,\n" 
  "                           global float *in_data,\n" 
  "                           {{index_type}} reductionSize,\n" 
  "                           {{index_type}} totalSlices,\n" 
  "                           float init) {\n" 
  "  const {{index_type}} sliceIndex = getReduceContigDimSliceIndex<{{index_type}}>();\n" 
  "\n" 
  "  if (sliceIndex >= totalSlices) {\n" 
  "    return;\n" 
  "  }\n" 
  "\n" 
  "  // Get the offset in `out` for the reduction\n" 
  "  const {{index_type}} outOffset =\n" 
  "    IndexToOffset_{{1000 + dim1}}_get(sliceIndex, out);\n" 
  "\n" 
  "  // Get the base offset in `in` for this block's reduction\n" 
  "  const {{index_type}} inBaseOffset =\n" 
  "    IndexToOffset_{{1000 + dim2}}_get(sliceIndex, in);\n" 
  "\n" 
  "  // Each thread in the block will reduce some subset of elements in\n" 
  "  // the slice. The elements are guaranteed contiguous starting at\n" 
  "  // `inBaseOffset`.\n" 
  "  float r = init;\n" 
  "  for ({{index_type}} i = threadIdx.x; i < reductionSize; i += blockDim.x) {\n" 
  "    r = reduceOp(r, modifyOp(in_data[inBaseOffset + i]));\n" 
  "  }\n" 
  "\n" 
  "  // Reduce within the block\n" 
  "  extern __shared__ float smem[];\n" 
  "  smem[threadIdx.x] = r;\n" 
  "\n" 
  "  // First warp will perform reductions across warps\n" 
  "  __syncthreads();\n" 
  "  if ((threadIdx.x / 32) == 0) {\n" 
  "    float r = {{init}};\n" 
  "    for (int i = 0; i < blockDim.x; i += 32) {\n" 
  "      r = reduceOp(r, smem[i + threadIdx.x]);\n" 
  "    }\n" 
  "\n" 
  "    // Each lane participating writes out a value\n" 
  "    smem[threadIdx.x] = r;\n" 
  "  }\n" 
  "\n" 
  "  // First thread will perform reductions across the block\n" 
  "  __syncthreads();\n" 
  "  if (threadIdx.x == 0) {\n" 
  "    r = init;\n" 
  "    #pragma unroll\n" 
  "    {% for 0=1,32 do %}\n" 
  "      r = reduceOp(r, smem[i]);\n" 
  "    {% end %}\n" 
  "\n" 
  "    // Write out reduced value\n" 
  "    out_data[outOffset] = r;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}


