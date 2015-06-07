#ifndef THCL_APPLY_INC
#define THCL_APPLY_INC

#include "THClTensorCopy.h"
#include "THClReduceApplyUtils.h"
#include "templates/TemplatedKernel.h"
#include "util/stringhelper.h"
#include "EasyCL.h"
#include "CLKernel_structs.h"

#include <string>

std::string getApply2_template();

//class Op2 {
//public:
//    std::string getOperation() = 0;
//};

//class Op3 {
//public:
//    std::string getOperation() = 0;
//};

//
// This file contains pointwise operation functions and kernels that
// work on both contiguous and non-contiguous tensor arguments of
// arbitrary (up to MAX_CLNN_DIMS) dimensioned arguments without
// copying or temporary storage.
//

// Threads per block for our apply kernel
#define THCL_APPLY_THREADS_PER_BLOCK 32 * 16

// Called when we are copying into an overlapping index `dst`, but
// we don't care which writer wins. Hacky but it works.
void THClTensor_copyIgnoringOverlaps(THClState* state,
                                       THClTensor* dst,
                                       THClTensor* src);

//template <typename Op, typename IndexType, int ADims>
//#if __CL_ARCH__ >= 350
//__launch_bounds__(32 * 16, 4)
//#endif
//__global__ void
//THClTensor_pointwiseApply1(TensorInfo<IndexType> a,
//                             IndexType totalElements,
//                             Op op) {
//  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
//       linearIndex < totalElements;
//       linearIndex += gridDim.x * blockDim.x) {
//    // Convert `linearIndex` into an offset of `a`
//    const IndexType aOffset =
//      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

//    op(&a.data[aOffset]);
//  }
//}

//template <typename Op, typename IndexType, int ADims, int BDims>
//#if __CL_ARCH__ >= 350
//__launch_bounds__(32 * 16, 4)
//#endif
//__global__ void
//THClTensor_pointwiseApply2(TensorInfo<IndexType> a,
//                             TensorInfo<IndexType> b,
//                             IndexType totalElements,
//                             Op op) {
//  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
//       linearIndex < totalElements;
//       linearIndex += gridDim.x * blockDim.x) {
//    // Convert `linearIndex` into an offset of `a`
//    const IndexType aOffset =
//      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

//    // Convert `linearIndex` into an offset of `b`
//    const IndexType bOffset =
//      IndexToOffset<IndexType, BDims>::get(linearIndex, b);

//    op(&a.data[aOffset], &b.data[bOffset]);
//  }
//}

typedef struct TensorInfoCl {
  TensorInfoCl( TensorInfo<unsigned int> info ) {
    dims = info.dims;
    if( info.offset > ( 1l << 30 ) ) {
      throw std::runtime_error("size " + toString(info.offset) + " out of bounds");
    }
    offset = (int)info.offset;
    for( int i = 0; i < dims; i++ ) {
      sizes[i] = info.sizes[i];
      strides[i] = info.strides[i];
    }
  }
  TensorInfoCl( TensorInfo<unsigned long> info ) {
    dims = info.dims;
    if( info.offset > ( 1l << 30 ) ) {
      throw std::runtime_error("size " + toString(info.offset) + " out of bounds");
    }
    offset = (int)info.offset;
    for( int i = 0; i < dims; i++ ) {
      if( info.sizes[i] > ( 1l << 31 ) ) {
        throw std::runtime_error("size " + toString(info.sizes[i]) + " out of bounds");
      }
      sizes[i] = info.sizes[i];
      strides[i] = info.strides[i];
    }
  }
  unsigned int sizes[MAX_CLNN_DIMS];
  unsigned int strides[MAX_CLNN_DIMS];
  int offset;
  int dims;
} TensorInfoCl;

template< typename Op, typename IndexType >
void kernelLaunch_pointwiseApply2( THClState *state, dim3 grid, dim3 block, int A, int B, TensorInfo<IndexType> aInfo, TensorInfo<IndexType> bInfo, IndexType totalElements, Op op ) {
    TemplatedKernel kernelBuilder( state->cl );
    kernelBuilder.set("adim", A);
    kernelBuilder.set("bdim", B);
    std::vector<int> dims;
    if( A >= 0 ) {
      dims.push_back(A);
    }
    if( B != A && B >= 0 ) {
        dims.push_back(B);
    }
    kernelBuilder.set("dims", dims);
    kernelBuilder.set("MAX_CLNN_DIMS", MAX_CLNN_DIMS);
    kernelBuilder.set("operation", op.operator2());
    std::string uniqueName = "apply2_" + toString(A) + "_" + toString(B) + "_" + op.operator2();
    CLKernel *kernel = kernelBuilder.buildKernel( uniqueName, "THClApply2.cl", getApply2_template(), "THClTensor_pointwiseApply2" );
    // calculate workgroup sizes and stuff
    dim3 global_ws;
    for( int i = 0; i < 3; i++ ) {
        global_ws.vec[i] = grid.vec[i] * block.vec[i];
    }

    // set up tensorinfos
    TensorInfoCl aInfoCl(aInfo);
    TensorInfoCl bInfoCl(bInfo);


    std::cout << "totalElements " << totalElements << std::endl;
    std::cout << "a offset " << aInfoCl.offset << 
        " b offset " << bInfoCl.offset << std::endl;
    std::cout << "adims " << aInfoCl.dims << " bdims " << bInfoCl.dims
        << std::endl;
    for( int i = 0; i < aInfoCl.dims; i++ ) {
        std::cout << "a dim" << i << " size=" << aInfoCl.sizes[i] << 
            " stride=" << aInfoCl.strides[i] << std::endl;
        std::cout << "b dim" << i << " size=" << bInfoCl.sizes[i] << 
            " stride=" << bInfoCl.strides[i] << std::endl;
    }
    std::cout<< "block " << block << std::endl;
    std::cout<< "grid " << grid << std::endl;
    std::cout<< "global_ws " << global_ws << std::endl;

    kernel->in(1, &aInfoCl)->in(1, &bInfoCl);
    kernel->inout( aInfo.wrapper )
          ->inout( bInfo.wrapper );
    if( totalElements > ( 1l << 30 )) {
        throw std::runtime_error("Error: out of bounds for totalelements=" + toString(totalElements));
    }
    kernel->in( (int)totalElements );
    kernel->run(3, global_ws.vec, block.vec);
    state->cl->finish();
}
//struct TensorInfo a,
//                             struct TensorInfo b,
//                            global float* a_data,
//                            global float*b_data,
//                             int totalElements


// this is a kernel, since marked with `__global__`
//template <typename Op, typename IndexType, int ADims, int BDims, int CDims>
//#if __CL_ARCH__ >= 350
//__launch_bounds__(32 * 16, 4)
//#endif
//__global__ void
//THClTensor_pointwiseApply3(TensorInfo<IndexType> a,
//                             TensorInfo<IndexType> b,
//                             TensorInfo<IndexType> c,
//                             IndexType totalElements,
//                             Op op) {
//  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
//       linearIndex < totalElements;
//       linearIndex += gridDim.x * blockDim.x) {
//    // Convert `linearIndex` into an offset of `a`
//    const IndexType aOffset =
//      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

//    // Convert `linearIndex` into an offset of `b`
//    const IndexType bOffset =
//      IndexToOffset<IndexType, BDims>::get(linearIndex, b);

//    // Convert `linearIndex` into an offset of `c`
//    const IndexType cOffset =
//      IndexToOffset<IndexType, CDims>::get(linearIndex, c);

//    op(&a.data[aOffset], &b.data[bOffset], &c.data[cOffset]);
//  }
//}

inline dim3 getApplyBlock() {
  return dim3(THCL_APPLY_THREADS_PER_BLOCK);
}

inline bool getApplyGrid(THClState* state, long totalElements, dim3& grid) {
//  int curDevice = -1;
//  cudaGetDevice(&curDevice);

//  if (curDevice == -1) {
//    return false;
//  }

//  // Assume a reasonable number of SMs if no state is available
//  int numSM =
//    state ? state->deviceProperties[curDevice].multiProcessorCount : 15;

  // dont think we can get number of SMs in OpenCL? (at least, not in opencl 1.1?)
  // just hardcode to 16 for now...
  // FIXME
  int numSM = 16;

  // 16 warps per block * 4 per SM gives 64 warps per SM at maximum,
  // which seems to be a good sweetspot for latency hiding
  grid = dim3(mymin(DIVUP(totalElements, (long long) THCL_APPLY_THREADS_PER_BLOCK),
                  4LL * numSM));
  return true;
}

//template <typename Op>
//bool THClTensor_pointwiseApply1(THClState* state,
//                                  THClTensor* a,
//                                  const Op& op,
//                                  TensorArgType aType = ReadWrite) {
//  long totalElements = THClTensor_nElement(state, a);

//  if (THClTensor_nDimension(state, a) > MAX_CLNN_DIMS) {
//    return false;
//  }

//  if (THClTensor_nDimension(state, a) == 0) {
//    // Zero-dim tensor; do nothing
//    return true;
//  }

//  const dim3 block = getApplyBlock();

//  dim3 grid;
//  if (!getApplyGrid(state, totalElements, grid)) {
//    return false;
//  }

//  // If tensor args have overlapping indices and are read/write, then
//  // we must expand the tensor to a contiguous form first, since
//  // otherwise there are conflicting writes. Upon copying back to the
//  // non-contiguous form, there will be conflicting writes, but at
//  // least with copy, one of the updaters will win atomically. This is
//  // a sketchy property of the old system as well (writing into all
//  // indices of a tensor with overlapping indices should probably be
//  // an error, since it is unclear which one should win), but we will
//  // preserve this last-writer-wins (in arbitrary copy order) behavior.
//  THClTensor* oldA = NULL;

//  if (aType == ReadWrite && THCL_overlappingIndices(state, a)) {
//    // Must perform in contiguous space
//    oldA = a;
//    a = THClTensor_newContiguous(state, a);
//  }

//  // It is possible that the tensor dimensions are able to be collapsed,
//  // and thus we can reduce the actual code complexity of the copy by
//  // exploiting this knowledge statically, since the div/mod is the
//  // most expensive part of the operation, more so than memory accesses.
//  // For instance, when copying a non-contiguous to a contiguous tensor
//  // (or vice versa), the contiguous tensor can be collapsed to one
//  // dimension, and the loop to translate the linear index to the array
//  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A)                                   \
  THError("Not implemented"); 
  /*THClTensor_pointwiseApply1<Op, TYPE, A>                    \
    <<<grid, block, 0, THClState_getCurrentStream(state)>>>(    \
      aInfo, (TYPE) totalElements, op);*/

#define HANDLE_A_CASE(TYPE, A)                      \
  {                                                 \
    if (aInfo.isContiguous()) {                     \
      HANDLE_CASE(TYPE, -2);                        \
    } else {                                        \
      switch (A) {                                  \
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

//  // Can we use 32-bit integer math in the kernel (the linear ID for the copy
//  // and the resulting non-linear offset is all computable using 32-bit math?)
//  // We also use unsigned index math in the kernel, as signed div/mod has
//  // additional overhead.
//  if (THCL_canUse32BitIndexMath(state, a)) {
//    TensorInfo<unsigned int> aInfo(state, a);

//    HANDLE_A_CASE(unsigned int, aInfo.dims);
//  } else {
//    TensorInfo<unsigned long> aInfo(state, a);

//    // For large tensors, we only compile the completely contiguous
//    // version and the completely generic version, to reduce
//    // compilation time.
//    if (aInfo.isContiguous()) {
//      THClTensor_pointwiseApply1<Op, unsigned long, -2>
//        <<<grid, block, 0, THClState_getCurrentStream(state)>>>(
//          aInfo, (unsigned long) totalElements, op);
//    } else {
//      THClTensor_pointwiseApply1<Op, unsigned long, -1>
//        <<<grid, block, 0, THClState_getCurrentStream(state)>>>(
//          aInfo, (unsigned long) totalElements, op);
//    }
//  }
#undef HANDLE_CASE
#undef HANDLE_A_CASE

//  if (oldA) {
//    // Ignore overlaps when copying back; if we use THClTensor_copy
//    // instead, it will recursively try and invoke ourselves to make
//    // oldA contiguous.
//    THClTensor_copyIgnoringOverlaps(state, oldA, a);
//    THClTensor_free(state, a);
//    a = oldA;
//  }

//  return true;
//}

template <typename Op>
bool THClTensor_pointwiseApply2(THClState* state,
                                  THClTensor* a,
                                  THClTensor* b,
                                  const Op& op,
                                  TensorArgType aType = ReadWrite,
                                  TensorArgType bType = ReadOnly) {
  long totalElements = THClTensor_nElement(state, a);

  if (totalElements != THClTensor_nElement(state, b)) {
    return false;
  }

  if (THClTensor_nDimension(state, a) > MAX_CLNN_DIMS ||
      THClTensor_nDimension(state, b) > MAX_CLNN_DIMS) {
    return false;
  }

  if (THClTensor_nDimension(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  THClTensor* oldA = NULL;
  THClTensor* oldB = NULL;

  if (aType == ReadWrite && THCL_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THClTensor_newContiguous(state, a);
  }
  if (bType == ReadWrite && THCL_overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = THClTensor_newContiguous(state, b);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A, B)                                \
   kernelLaunch_pointwiseApply2<Op, TYPE>(state, grid, block, A, B, aInfo, bInfo, (TYPE) totalElements, op ); \
  /* THClTensor_pointwiseApply2<Op, TYPE, A, B>                 \
    <<<grid, block, 0, THClState_getCurrentStream(state)>>>(    \
      aInfo, bInfo, (TYPE) totalElements, op); */

#define HANDLE_B_CASE(TYPE, A, B)                   \
  {                                                 \
    if (bInfo.isContiguous()) {                     \
      HANDLE_CASE(TYPE, A, -2);                     \
    } else {                                        \
      switch (B) {                                  \
        case 1:                                     \
          HANDLE_CASE(TYPE, A, 1);                  \
          break;                                    \
        case 2:                                     \
          HANDLE_CASE(TYPE, A, 2);                  \
          break;                                    \
        case 3:                                     \
          HANDLE_CASE(TYPE, A, 3);                  \
          break;                                    \
        default:                                    \
          HANDLE_CASE(TYPE, A, -1);                 \
          break;                                    \
      }                                             \
    }                                               \
  }

#define HANDLE_A_CASE(TYPE, A, B)                   \
  {                                                 \
    if (aInfo.isContiguous()) {                     \
      HANDLE_B_CASE(TYPE, -2, B);                   \
    } else {                                        \
      switch (A) {                                  \
        case 1:                                     \
          HANDLE_B_CASE(TYPE, 1, B);                \
          break;                                    \
        case 2:                                     \
          HANDLE_B_CASE(TYPE, 2, B);                \
          break;                                    \
        case 3:                                     \
          HANDLE_B_CASE(TYPE, 3, B);                \
          break;                                    \
        default:                                    \
          HANDLE_B_CASE(TYPE, -1, B);               \
          break;                                    \
      }                                             \
    }                                               \
  }

  if (THCL_canUse32BitIndexMath(state, a) &&
      THCL_canUse32BitIndexMath(state, b)) {
    TensorInfo<unsigned int> aInfo(state, a);
    TensorInfo<unsigned int> bInfo(state, b);

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);
    TensorInfo<unsigned long> bInfo(state, b);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous()) {
      THError("Not implemented");
//      THClTensor_pointwiseApply2<Op, unsigned long, -2, -2>
//        <<<grid, block, 0, THClState_getCurrentStream(state)>>>(
//          aInfo, bInfo, (unsigned long) totalElements, op);
    } else {
      THError("Not implemented");
//      THClTensor_pointwiseApply2<Op, unsigned long, -1, -1>
//        <<<grid, block, 0, THClState_getCurrentStream(state)>>>(
//          aInfo, bInfo, (unsigned long) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THClTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THClTensor_copyIgnoringOverlaps(state, oldA, a);
    THClTensor_free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THClTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    THClTensor_copyIgnoringOverlaps(state, oldB, b);
    THClTensor_free(state, b);
    b = oldB;
  }

  return true;
}

// this is a c++ template
template <typename Op>
bool THClTensor_pointwiseApply3(THClState* state,
                                  THClTensor* a,
                                  THClTensor* b,
                                  THClTensor* c,
                                  const Op& op,
                                  TensorArgType aType = ReadWrite,
                                  TensorArgType bType = ReadOnly,
                                  TensorArgType cType = ReadOnly) {
  long totalElements = THClTensor_nElement(state, a);

  if (totalElements != THClTensor_nElement(state, b) ||
      totalElements != THClTensor_nElement(state, c)) {
    return false;
  }

  if (THClTensor_nDimension(state, a) > MAX_CLNN_DIMS ||
      THClTensor_nDimension(state, b) > MAX_CLNN_DIMS ||
      THClTensor_nDimension(state, c) > MAX_CLNN_DIMS) {
    return false;
  }

  if (THClTensor_nDimension(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  THClTensor* oldA = NULL;
  THClTensor* oldB = NULL;
  THClTensor* oldC = NULL;

  if (aType == ReadWrite && THCL_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THClTensor_newContiguous(state, a);
  }

  if (bType == ReadWrite && THCL_overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = THClTensor_newContiguous(state, b);
  }

  if (cType == ReadWrite && THCL_overlappingIndices(state, c)) {
    // Must perform in contiguous space
    oldC = c;
    c = THClTensor_newContiguous(state, c);
  }

#define HANDLE_CASE(TYPE, A, B, C)                                      \
  THError("Not implemented");  
    /* kernel launch ... */ \
  /* THClTensor_pointwiseApply3<Op, TYPE, A, B, C> */                      \
    /* <<<grid, block, 0, THClState_getCurrentStream(state)>>>(             \
      aInfo, bInfo, cInfo, (TYPE) totalElements, op); */

#define HANDLE_C_CASE(TYPE, A, B, C)             \
  {                                              \
    if (cInfo.isContiguous()) {                  \
      HANDLE_CASE(TYPE, A, B, -2);               \
    } else {                                     \
      switch (C) {                               \
        case 1:                                  \
          HANDLE_CASE(TYPE, A, B, 1);            \
          break;                                 \
        case 2:                                  \
          HANDLE_CASE(TYPE, A, B, 2);            \
          break;                                 \
        case 3:                                  \
          HANDLE_CASE(TYPE, A, B, 3);            \
          break;                                 \
        default:                                 \
          HANDLE_CASE(TYPE, A, B, -1);           \
          break;                                 \
      }                                          \
    }                                            \
  }

#define HANDLE_B_CASE(TYPE, A, B, C)                 \
  {                                                  \
    if (bInfo.isContiguous()) {                      \
      HANDLE_C_CASE(TYPE, A, -2, C);                 \
    } else {                                         \
      switch (B) {                                   \
        case 1:                                      \
          HANDLE_C_CASE(TYPE, A, 1, C);              \
          break;                                     \
        case 2:                                      \
          HANDLE_C_CASE(TYPE, A, 2, C);              \
          break;                                     \
        case 3:                                      \
          HANDLE_C_CASE(TYPE, A, 3, C);              \
          break;                                     \
        default:                                     \
          HANDLE_C_CASE(TYPE, A, -1, C);             \
          break;                                     \
      }                                              \
    }                                                \
  }

#define HANDLE_A_CASE(TYPE, A, B, C)                 \
  {                                                  \
    if (aInfo.isContiguous()) {                      \
      HANDLE_B_CASE(TYPE, -2, B, C);                 \
    } else {                                         \
      switch (A) {                                   \
        case 1:                                      \
          HANDLE_B_CASE(TYPE, 1, B, C);              \
          break;                                     \
        case 2:                                      \
          HANDLE_B_CASE(TYPE, 2, B, C);              \
          break;                                     \
        case 3:                                      \
          HANDLE_B_CASE(TYPE, 3, B, C);              \
          break;                                     \
        default:                                     \
          HANDLE_B_CASE(TYPE, -1, B, C);             \
          break;                                     \
      }                                              \
    }                                                \
  }

  if (THCL_canUse32BitIndexMath(state, a) &&
      THCL_canUse32BitIndexMath(state, b) &&
      THCL_canUse32BitIndexMath(state, c)) {
    TensorInfo<unsigned int> aInfo(state, a);
    TensorInfo<unsigned int> bInfo(state, b);
    TensorInfo<unsigned int> cInfo(state, c);

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims, cInfo.dims);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);
    TensorInfo<unsigned long> bInfo(state, b);
    TensorInfo<unsigned long> cInfo(state, c);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous() && cInfo.isContiguous()) {
      THError("Not implemented");
//      THClTensor_pointwiseApply3<Op, unsigned long, -2, -2, -2>
//        <<<grid, block, 0, THClState_getCurrentStream(state)>>>(
//          aInfo, bInfo, cInfo, (unsigned long) totalElements, op);
    } else {
      THError("Not implemented");
//      THClTensor_pointwiseApply3<Op, unsigned long, -1, -1, -1>
//        <<<grid, block, 0, THClState_getCurrentStream(state)>>>(
//          aInfo, bInfo, cInfo, (unsigned long) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_C_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THClTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THClTensor_copyIgnoringOverlaps(state, oldA, a);
    THClTensor_free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THClTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    THClTensor_copyIgnoringOverlaps(state, oldB, b);
    THClTensor_free(state, b);
    b = oldB;
  }

  if (oldC) {
    // Ignore overlaps when copying back; if we use THClTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    THClTensor_copyIgnoringOverlaps(state, oldC, c);
    THClTensor_free(state, c);
    c = oldC;
  }

  return true;
}

#undef THCL_APPLY_THREADS_PER_BLOCK

#endif // THCL_APPLY_INC

