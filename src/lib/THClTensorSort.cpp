// from lib/THC/THCTensorSort.cu:

#include "THClReduceApplyUtils.cuh"
#include "THClSortUtils.cuh"
#include "THClTensorCopy.h"

#include "EasyCL.h"
#include "CLKernel_structs.h"
#include "util/easycl_stringhelper.h"
#include "util/StatefulTimer.h"
#include "templates/TemplatedKernel.h"

#include <string>
#include <iostream>

using namespace std;

static std::string get_template();

// Returns 2^(ceil(lg(n)) from Stanford bit twiddling hacks
unsigned long nextHighestPowerOf2(unsigned long n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  n++;

  return n;
}

static template< typename IndexType >
void kernelLaunch_fillSliceWithIndex(
                     THClState* state,
                     dim3 &grid, dim3 &block, size_t smemSize,
                     int ADims,
                     const TensorInfo<IndexType> & in,
                     int64 totalElements,
                     CLWrapper* scratch
    ){
  StatefulTimer::timeCheck("fillSliceWithIndex START");
  std::string uniqueName = "THClTensorSort_fillSliceWithIndex_" + easycl::toString(ADims) + "_" + modifyOp->operator2() + "_" + reduceOp->operator3();
  EasyCL *cl = scratch->getCl();
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("fillSliceWithIndex 1aa");
  } else {
    std::vector< int > dims;
    if( ADims >= 0 ) {
      dims.push_back(ADims);
    }
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder
//      .set("include_THClDeviceUtils", THClDeviceUtils_getKernelTemplate())
      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
//      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
      .set("dims", dims)
      .set("dim1", ADims)
//      .set("defreduceblock", 1)
//      .set("modify_operation", modifyOp->operator2())
//      .set("reduce_operation", reduceOp->operator3())
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
      .set("IndexType", TypeParseTraits<IndexType>::name)
    ;
    kernel = kernelBuilder.buildKernel( uniqueName, "THClTensorSort.cl", getKernelTemplate(), "fillSliceWithIndex" );
  }

  THClKernels k(state, kernel);
  k.in(in);
  k.in((int)totalElements);
  k.in(init);
  k.out(scratch);
  k.localFloats(smemSize / sizeof(float));
  k.run(grid, block);
  if(state->addFinish) cl->finish();  
  StatefulTimer::timeCheck("fillSliceWithIndex END");
}

void THClTensor_fillSliceWithIndex(THClState* state,
                                     THClTensor* t,
                                     int dim) {
  THCCheckTensorDims(state, t, 2);

  long inElements = THClTensor_nElement(state, t);
  long sliceSize = THClTensor_size(state, t, dim);
  long numSlices = inElements / sliceSize;

  dim3 grid;
  if (!THCL_getGridFromTiles(numSlices, grid)) {
    THError("Slice to fill with indices is too large");
  }

  long maxThreads =
    THClState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  long numThreads = sliceSize;
  if (numThreads > maxThreads) {
    numThreads = maxThreads;
  }

  dim3 block(numThreads);

#define FILL_INDEX(T, DIM)                                       \
  kernelLaunch_fillSliceWithIndex<T>(                                     \
      grid, block, 0, info, numSlices, sliceSize, info.strides[collapseDim])

  if (THCL_canUse32BitIndexMath(state, t)) {
    TensorInfo<unsigned int> info(state, t, dim);
    info.sizes[dim] = 1;
    int collapseDim = info.collapseDims(dim);

    if (info.isContiguous()) {
      FILL_INDEX(unsigned int, -2);
    } else {
      if (info.dims == 1) {
        FILL_INDEX(unsigned int, 1);
      } else if (info.dims == 2) {
        FILL_INDEX(unsigned int, 2);
      } else {
        FILL_INDEX(unsigned int, -1);
      }
    }
  } else {
    TensorInfo<unsigned long> info(state, t, dim);
    info.sizes[dim] = 1;
    int collapseDim = info.collapseDims(dim);

    // catch-all implementation
    FILL_INDEX(unsigned long, -1);
  }

#undef FILL_INDEX


}

THCL_API void THClTensor_sortKeyValueInplace(THClState* state,
                                              THClTensor* key,
                                              THClTensor* value,
                                              int dim, bool dir) {
  THArgCheck(THClTensor_isSameSizeAs(state, key, value), 2,
             "Key tensor must have same size as value tensor");
  THCCheckTensorDims(state, key, 2);
  THCCheckTensorDims(state, value, 3);

  long inElements = THClTensor_nElement(state, key);
  long keySliceSize = THClTensor_size(state, key, dim);
  long keySlices = inElements / keySliceSize;

  if (THClTensor_nDimension(state, key) == 0) {
    // Zero-dim tensor; do nothing
    return;
  }

  // The amount of shared memory and block size is based on
  // 2^ceil(lg(n)); we choose that sorting implementation for a given
  // size.
  long ceilPowerOf2 = nextHighestPowerOf2(keySliceSize);

  // FIXME: We'd have to find some other trick with Thrust to perform a
  // vectorized (key, value) sort by slice segment
  if (ceilPowerOf2 > 2048) {
    THError("sortKeyValueInplace only works for sizes <= 2048 at present");
  }

  int blockSize = (int) ceilPowerOf2 / 2;
  if (blockSize < 1) {
    blockSize = 1;
  }

  dim3 block(blockSize);

  // The grid is based on the number of independent slices that we
  // have to sort; one block per slice
  dim3 grid;
  if (!THCL_getGridFromTiles(keySlices, grid)) {
    THError("Slice to sort is too large");
  }

#define HANDLE_CASE(TYPE, A, SIZE)                                      \
  if (dir) {                                                            \
    bitonicSortKVInPlace<float, float, A, -1, GTComp<float>, TYPE, SIZE> \
      <<<grid, block, 0, THClState_getCurrentStream(state)>>>(           \
        keyInfo,                                                        \
        keySlices,                                                      \
        (TYPE) keySliceSize,                                            \
        (TYPE) keyInfo.strides[collapseKeyDim],                         \
        valueInfo,                                                      \
        (TYPE) valueInfo.strides[collapseValueDim],                     \
        GTComp<float>());                                               \
  } else {                                                              \
    bitonicSortKVInPlace<float, float, A, -1, LTComp<float>, TYPE, SIZE> \
      <<<grid, block, 0, THClState_getCurrentStream(state)>>>(           \
        keyInfo,                                                        \
        keySlices,                                                      \
        (TYPE) keySliceSize,                                            \
        (TYPE) keyInfo.strides[collapseKeyDim],                         \
        valueInfo,                                                      \
        (TYPE) valueInfo.strides[collapseValueDim],                     \
        LTComp<float>());                                               \
  }

#define HANDLE_SORT_CASE(TYPE, A)                       \
  {                                                     \
    switch (ceilPowerOf2) {                             \
      case 2048:                                        \
      HANDLE_CASE(TYPE, A, 2048);                       \
      break;                                            \
      case 1024:                                        \
      HANDLE_CASE(TYPE, A, 1024);                       \
      break;                                            \
      case 512:                                         \
      HANDLE_CASE(TYPE, A, 512);                        \
      break;                                            \
      case 256:                                         \
      HANDLE_CASE(TYPE, A, 256);                        \
      break;                                            \
      case 128:                                         \
      HANDLE_CASE(TYPE, A, 128);                        \
      break;                                            \
      case 64:                                          \
      HANDLE_CASE(TYPE, A, 64);                         \
      break;                                            \
      case 32:                                          \
      HANDLE_CASE(TYPE, A, 32);                         \
      break;                                            \
      case 16:                                          \
      HANDLE_CASE(TYPE, A, 16);                         \
      break;                                            \
      case 8:                                           \
      HANDLE_CASE(TYPE, A, 8);                          \
      break;                                            \
      case 4:                                           \
      HANDLE_CASE(TYPE, A, 4);                          \
      break;                                            \
      case 2:                                           \
      HANDLE_CASE(TYPE, A, 2);                          \
      break;                                            \
      case 1:                                           \
      /* Nothing to do, data already sorted */          \
      break;                                            \
      default:                                          \
      assert(false);                                    \
    }                                                   \
  }

  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  if (THCL_canUse32BitIndexMath(state, key)) {
    TensorInfo<unsigned int> keyInfo(state, key);
    keyInfo.sizes[dim] = 1;
    int collapseKeyDim = keyInfo.collapseDims(dim);

    TensorInfo<unsigned int> valueInfo(state, value);
    valueInfo.sizes[dim] = 1;
    int collapseValueDim = valueInfo.collapseDims(dim);

    if (keyInfo.isContiguous()) {
      HANDLE_SORT_CASE(unsigned int, -2);
    } else {
      switch (keyInfo.dims) {
        case 1:
          HANDLE_SORT_CASE(unsigned int, 1);
          break;
        case 2:
          HANDLE_SORT_CASE(unsigned int, 2);
          break;
        default:
          HANDLE_SORT_CASE(unsigned int, -1);
          break;
      }
    }
  } else {
    TensorInfo<unsigned long> keyInfo(state, key);
    keyInfo.sizes[dim] = 1;
    int collapseKeyDim = keyInfo.collapseDims(dim);

    TensorInfo<unsigned long> valueInfo(state, value);
    valueInfo.sizes[dim] = 1;
    int collapseValueDim = valueInfo.collapseDims(dim);

    // long case is rare, just instantiate these versions
    if (keyInfo.isContiguous()) {
      HANDLE_SORT_CASE(unsigned long, -2);
    } else {
      HANDLE_SORT_CASE(unsigned long, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_SORT_CASE
#undef HANDLE_A_CASE


}

THCL_API void THClTensor_sort(THClState* state,
                               THClTensor *sorted,
                               THClTensor *indices,
                               THClTensor *input,
                               int dim, int order) {
  THAssert(THClTensor_checkGPU(state, 3, sorted, indices, input));
  THCCheckTensorDims(state, sorted, 2);
  THCCheckTensorDims(state, indices, 3);
  THCCheckTensorDims(state, input, 4);

  // Make sure sufficient output space is allocated
  THClTensor_resizeAs(state, sorted, input);
  THClTensor_resizeAs(state, indices, input);

  // How large are the slices that we are sorting?
  long totalElements = THClTensor_nElement(state, input);
  long sliceSize = THClTensor_size(state, input, dim);

  // We're using THClTensor to write out indices, so if the slice
  // size that we're sorting has more elements than can be
  // represented in fp32, warn the user
  // FIXME: this isn't a real restriction of either our code or of
  // Thrust, but we have to switch to a CL long tensor to support
  // larger slice sizes. Otherwise the indices will contain garbage.
  THArgCheck(sliceSize <= (long) FLOAT32_MAX_CONSECUTIVE_INT, 5,
             "The sort dimension exceeds single-precision float "
             "consecutive integer precision size (2^24), since float "
             "is used for indices");

  if (sliceSize <= 2048) {
    // Fill `indices` (the values) with the
    // slice-relative index.
    THClTensor_fillSliceWithIndex(state, indices, dim);

    // We sort k/v pairs in-place; copy unsorted input to output
    THClTensor_copy(state, sorted, input);

    // Sort using our in-place k/v kernel that supports arbitrary
    // layout
    THClTensor_sortKeyValueInplace(state, sorted, indices, dim, order);
  } else {
    // Otherwise, fall back upon Thrust, which handles all other cases
    // (potentially slowly, with extra copies/memory allocations)
    THError("sort not implemented for slice size > 2048");
  }
}

std::string get_template() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClTensorSort.cl" )
  // ]]]
  // generated using cog, from THClTensorSort.cl:
  const char * kernelSource =  
  "// from lib/THC/THCTensorSort.cu:\n" 
  "\n" 
  "// needs following tmeplate variables defined:\n" 
  "//  Dim      integer\n" 
  "//  IndexType  string 'int'\n" 
  "\n" 
  "{{include_THClReduceApplyUtils}}\n" 
  "\n" 
  "{{include_THClSortUtils}}\n" 
  "\n" 
  "// `base` is the base address of a tensor\n" 
  "// For each slice (defined as a linear point of `out`, from 0 ->\n" 
  "// (sliceSize - 1) * sliceStride, we fill that slice from `0` to\n" 
  "// `sliceSize - 1`.\n" 
  "kernel void\n" 
  "fillSliceWithIndex(global TensorInfoCl *out_info, global float *out_data,\n" 
  "                   {{IndexType}} totalSlices,\n" 
  "                   {{IndexType}} sliceSize,\n" 
  "                   {{IndexType}} sliceStride) {\n" 
  "  {{IndexType}} slice = getLinearBlockId();\n" 
  "\n" 
  "  if (slice >= totalSlices) {\n" 
  "    return;\n" 
  "  }\n" 
  "\n" 
  "  const unsigned long offset =\n" 
  "    IndexToOffset_{{1000+Dim}}_get(slice, &out_info[0]);\n" 
  "  float* base = &out_data[offset];\n" 
  "\n" 
  "  for (long i = get_local_id(0); i < sliceSize; i += get_local_size(0)) {\n" 
  "    // Torch indices are 1-based (hence the +1)\n" 
  "    base[i * sliceStride] = (float) i + 1.0f;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

