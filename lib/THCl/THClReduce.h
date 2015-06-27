#pragma once

#ifndef THCL_REDUCE_INC
#define THCL_REDUCE_INC

#include <string>
#include <vector>
#include <set>
#include "THClReduceApplyUtils.h"
#include "templates/TemplatedKernel.h"
#include "util/easycl_stringhelper.h"
#include "EasyCL.h"
#include "THClTypeParseTraits.h"
#include "THClDeviceUtils.h"
#include "THClKernels.h"
#include "util/StatefulTimer.h"


std::string THClReduce_getKernelSource();

//
// This file contains dimension reduction operation functions and
// kernels that work on both contiguous and non-contiguous tensor
// arguments of arbitrary (up to MAX_CUTORCH_DIMS) dimensioned
// arguments without copying or temporary storage.
//

//#define THCL_NONCONTIG_REDUCE_BLOCK_SIZE 32 * 16

inline long getNonContigReduceBlockSize(THClState *state) {
  int blockSize = 1024;
  int maxWorkgroupSize = ((easycl::DeviceInfo *)state->deviceInfoByDevice[state->currentDevice])->maxWorkGroupSize;
  if( blockSize > maxWorkgroupSize ) {
    blockSize = maxWorkgroupSize;
  }
  return blockSize;
}

inline dim3 getNoncontigReduceBlock(THClState *state) {
  return dim3(getNonContigReduceBlockSize(state));
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
  long warpsInReductionSize = THClCeilDiv(reductionSize, 32L);
//  long warpsInReductionSize = DIVUP(reductionSize, 32L);
  int numWarps =
    warpsInReductionSize > (long) maxWarps ? maxWarps : (int) warpsInReductionSize;
//    warpsInReductionSize > (long) maxWarps ? maxWarps : (int) warpsInReductionSize;
  return dim3(numWarps * 32);
}

inline bool getNoncontigReduceGrid(THClState *state, long elements, dim3& grid) {
  // One output point per thread
  return THCL_getGridFromTiles(THClCeilDiv(elements, (long) getNonContigReduceBlockSize(state)), grid);
//  return THCL_getGridFromTiles(DIVUP(elements, THCL_NONCONTIG_REDUCE_BLOCK_SIZE), grid);
}

inline bool getContigReduceGrid(long elements, dim3& grid) {
  // One output point per block
  return THCL_getGridFromTiles(elements, grid);
}

template<typename IndexType>
void kernelLaunch_THClTensor_reduceNoncontigDim(
  THClState *state,
  dim3 &grid,
  dim3 &block,
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

//  THClState_getCl(state)->finish();
  StatefulTimer::timeCheck("Reduce-noncontig START");
  std::string uniqueName = "THClTensor_reduceNoncontigDim_" + easycl::toString(ADims) + "_" + easycl::toString(BDims) + "_" +
    TypeParseTraits<IndexType>::name + "_" + modifyOp->operator2() + "_" + reduceOp->operator3();
  EasyCL *cl = THClState_getCl(state);
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("Apply3 1aa");
  } else {
    // launch kernel here....
    TemplatedKernel kernelBuilder(THClState_getCl(state));

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

    std::string indexType = TypeParseTraits<IndexType>::name;
  //  indexType = easycl::replace(indexType, " ", "");
    kernelBuilder
      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
      .set("dims", dims)
      .set("dim1", ADims)
      .set("dim2", BDims)
      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
      .set("IndexType", indexType)
      .set("modify_operation", modifyOp->operator2())
      .set("reduce_operation", reduceOp->operator3())
    ;

    kernel = kernelBuilder.buildKernel(uniqueName, "THClReduce.cl", THClReduce_getKernelSource(), "THClTensor_reduceNoncontigDim");
  }

  THClKernels k(state, kernel);
  k.out(out);
  k.in(in);
  k.in((int)reductionStride);
  k.in((int)reductionSize);
  k.in((int)totalSlices);
  k.in(init);

  k.run(grid, block);

  THClState_getCl(state)->finish();  
  StatefulTimer::timeCheck("Reduce-noncontig END");
}

template<typename IndexType>
void kernelLaunch_THClTensor_reduceContigDim(
  THClState *state,
  dim3 &grid,
  dim3 &block,
  size_t smemSize,
  int ADims,
  int BDims,
  TensorInfo<IndexType> out,
  TensorInfo<IndexType> in,
  IndexType reductionSize,
  IndexType totalSlices,
  float init,
  HasOperator2 const*modifyOp,
  HasOperator3 const*reduceOp) {

//  THClState_getCl(state)->finish();
  StatefulTimer::timeCheck("Reduce-contig START");
  std::string uniqueName = "THClTensor_reduceContigDim_" + easycl::toString(ADims) + "_" + easycl::toString(BDims) + "_" +
    TypeParseTraits<IndexType>::name + "_" + modifyOp->operator2() + "_" + reduceOp->operator3();
  EasyCL *cl = THClState_getCl(state);
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("Apply3 1aa");
  } else {

    // launch kernel here....

    TemplatedKernel kernelBuilder(THClState_getCl(state));

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

    std::string indexType = TypeParseTraits<IndexType>::name;
  //  indexType = easycl::replace(indexType, " ", "");
    kernelBuilder
      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
      .set("dims", dims)
      .set("dim1", ADims)
      .set("dim2", BDims)
      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
  //    .set("IndexType", indexType)
      .set("IndexType", "int")
      .set("modify_operation", modifyOp->operator2())
      .set("reduce_operation", reduceOp->operator3())
    ;

    kernel = kernelBuilder.buildKernel(uniqueName, "THClReduce.cl", THClReduce_getKernelSource(), "THClTensor_reduceContigDim");
  }

  THClKernels k(state, kernel);
  k.out(out);
  k.in(in);
  k.in((int)reductionSize);
  k.in((int)totalSlices);
  k.in(init);
  k.localFloats(smemSize / sizeof(float));

  k.run(grid, block);

  THClState_getCl(state)->finish();
  StatefulTimer::timeCheck("Reduce-config END");
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

