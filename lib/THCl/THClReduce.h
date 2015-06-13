#pragma once

#ifndef THCL_REDUCE_INC
#define THCL_REDUCE_INC

#include <string>
#include <vector>
#include <set>
#include "THClTensorInfoCl.h"
#include "templates/TemplatedKernel.h"
#include "util/easycl_stringhelper.h"
#include "EasyCL.h"

std::string getKernelSource();


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

template<typename IndexType>
void kernelLaunch_THClTensor_reduceNoncontigDim(
  THClState *state,
  int ADims,
  int BDims,
  TensorInfo<IndexType> out,
  TensorInfo<IndexType> in,
  IndexType reductionStride,
  IndexType reductionSize,
  IndexType totalSlices,
  float init,
  HasOperator2 const*modifyOp,
  HasOperator3 const*reduceOp) {

  // launch kernel here....
  TemplatedKernel kernelBuilder(state->cl);

  std::set<int> dims_set;
  if(ADims >= 0) {
    dims_set.insert(ADims);
  }
  if(BDims >= 0) {
    dims_set.insert(BDims);
  }
  std::vector<int> dims;
  for( std::set<int>::iterator it = dims_set.begin(); it != dims_set.end(); it++ ) {
    dims.push_back(*it);
  }

  kernelBuilder
    .set("include_TensorInfoCl", THClTensorInfoCl_getKernelTemplate())
    .set("dims", dims)
    .set("dim1", ADims)
    .set("dim2", BDims)
    .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
    .set("index_type", TypeParseTraits<IndexType>::name)
    .set("modify_operation", modifyOp->operator2())
    .set("reduce_operation", reduceOp->operator3())
  ;

  std::string uniqueName = "THClTensor_reduceNoncontigDim_" + easycl::toString(ADims) + "_" + easycl::toString(BDims) + "_" +
    TypeParseTraits<IndexType>::name + "_" + modifyOp->operator2() + "_" + reduceOp->operator3();
  CLKernel *kernel = kernelBuilder.buildKernel(uniqueName, "THClReduce.cl", getKernelSource(), "THClTensor_reduceNoncontigDim");
//  kernel->in();

  THError("Not implemented");
}

template<typename IndexType>
void kernelLaunch_THClTensor_reduceContigDim(
  THClState *state,
  int ADims,
  int BDims,
  TensorInfo<IndexType> out,
  TensorInfo<IndexType> in,
  IndexType reductionSize,
  IndexType totalSlices,
  float init,
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
                          float init,
                          const HasOperator2 *modifyOp,
                          const HasOperator3 *reduceOp,
                          int dim);

#undef THCL_NONCONTIG_REDUCE_BLOCK_SIZE

#endif // THCL_REDUCE_INC

