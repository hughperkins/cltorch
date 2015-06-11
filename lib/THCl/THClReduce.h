#ifndef THCL_REDUCE_INC
#define THCL_REDUCE_INC

//
// This file contains dimension reduction operation functions and
// kernels that work on both contiguous and non-contiguous tensor
// arguments of arbitrary (up to MAX_CUTORCH_DIMS) dimensioned
// arguments without copying or temporary storage.
//

#include "THClReduceApplyUtils.h"

#define THCL_NONCONTIG_REDUCE_BLOCK_SIZE 32 * 16

inline dim3 getNoncontigReduceBlock() {
  return dim3(THCL_NONCONTIG_REDUCE_BLOCK_SIZE);
}


template<typename IndexType, int ADims, int BDims>
void kernelLaunch_THClTensor_reduceNoncontigDim(
  TensorInfo<IndexType> out,
  TensorInfo<IndexType> in,
  IndexType reductionStride,
  IndexType reductionSize,
  IndexType totalSlices,
  HasOperator2 const*modifyOp,
  HasOperator3 const*reduceOp) {

  // launch kernel here....

  THError("Not implemented");
}

template<typename IndexType, int ADims, int BDims>
void kernelLaunch_THClTensor_reduceContigDim(
  TensorInfo<IndexType> out,
  TensorInfo<IndexType> in,
  IndexType reductionSize,
  IndexType totalSlices,
  HasOperator2 const*modifyOp,
  HasOperator3 const*reduceOp) {

  // launch kernel here....

  THError("Not implemented");
}

inline dim3 getContigReduceBlock(long numSlices, long reductionSize) {
  // If the number of slices is low but the reduction dimension size
  // is high, then we should increase block size for greater parallelism.
  // Aim for at least 32 warps per SM (assume 15 SMs; don't bother
  // inquiring the real number for now).
  int maxWarps = 4; // better occupancy if many blocks are around
  // For numSlices > 15 * 8, there are > 32 warps active per SM.
  if (numSlices < 15 * 8) {
    maxWarps = 8;
    if (numSlices < 15 * 4) {
      maxWarps = 16;
      if (numSlices < 15 * 2) {
        maxWarps = 32;
      }
    }
  }

  // Scale up block size based on the reduction dimension size
  long warpsInReductionSize = DIVUP(reductionSize, 32L);
  int numWarps =
    warpsInReductionSize > (long) maxWarps ? maxWarps : (int) warpsInReductionSize;
  return dim3(numWarps * 32);
}

inline bool getNoncontigReduceGrid(long elements, dim3& grid) {
  // One output point per thread
  return THCL_getGridFromTiles(DIVUP(elements, THCL_NONCONTIG_REDUCE_BLOCK_SIZE), grid);
}

inline bool getContigReduceGrid(long elements, dim3& grid) {
  // One output point per block
  return THCL_getGridFromTiles(elements, grid);
}

bool THClTensor_reduceDim(THClState* state,
                          THClTensor* out,
                          THClTensor* in,
                          const HasOperator1 *modifyOp,
                          const HasOperator2 *reduceOp,
                          int dim);

#undef THCL_NONCONTIG_REDUCE_BLOCK_SIZE

#endif // THCL_REDUCE_INC

