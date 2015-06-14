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


std::string THClReduce_getKernelSource();

//
// This file contains dimension reduction operation functions and
// kernels that work on both contiguous and non-contiguous tensor
// arguments of arbitrary (up to MAX_CUTORCH_DIMS) dimensioned
// arguments without copying or temporary storage.
//

#define THCL_NONCONTIG_REDUCE_BLOCK_SIZE 32 * 16

inline dim3 getNoncontigReduceBlock() {
  return dim3(THCL_NONCONTIG_REDUCE_BLOCK_SIZE);
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

inline bool getNoncontigReduceGrid(long elements, dim3& grid) {
  // One output point per thread
  return THCL_getGridFromTiles(THClCeilDiv(elements, (long) THCL_NONCONTIG_REDUCE_BLOCK_SIZE), grid);
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

  std::string uniqueName = "THClTensor_reduceNoncontigDim_" + easycl::toString(ADims) + "_" + easycl::toString(BDims) + "_" +
    TypeParseTraits<IndexType>::name + "_" + modifyOp->operator2() + "_" + reduceOp->operator3();
  CLKernel *kernel = kernelBuilder.buildKernel(uniqueName, "THClReduce.cl", THClReduce_getKernelSource(), "THClTensor_reduceNoncontigDim");
  // calculate workgroup sizes and stuff
  dim3 global_ws;
  for( int i = 0; i < 3; i++ ) {
      global_ws.vec[i] = grid.vec[i] * block.vec[i];
  }

  // set up tensorinfos
  TensorInfoCl outCl(out);
  TensorInfoCl inCl(in);

  if( !out.wrapper->isOnDevice() ) {
    out.wrapper->createOnDevice();
  }
  // debugging
  in.wrapper->copyToHost();
  for( int i = 0; i < 3; i++ ) {
    float *indata = (float *)in.wrapper->getHostArray();
    std::cout << "in[" << i << "]=" << indata[i] << std::endl;
  }

  kernel->in(1, &outCl);
  kernel->out( out.wrapper );
  kernel->in(1, &inCl);
  kernel->in( in.wrapper );
  kernel->in((int)reductionStride);
  kernel->in((int)reductionSize);
  kernel->in((int)totalSlices);
  kernel->in(init);

  std::cout << "reductionstride " << reductionStride << std::endl;
  std::cout << "reductionsize " << reductionSize << std::endl;
  std::cout << "totalSlices " << totalSlices << std::endl;
  std::cout << "grid " << grid << std::endl;
  std::cout << "block " << block << std::endl;

  kernel->run(3, global_ws.vec, block.vec);
  THClState_getCl(state)->finish();

  // debugging
  out.wrapper->copyToHost();
  for( int i = 0; i < 3; i++ ) {
    float *outdata = (float *)out.wrapper->getHostArray();
    std::cout << "out[" << i << "]=" << outdata[i] << std::endl;
  }

//  THError("Not implemented");
}

//THClTensor_reduceNoncontigDim(global TensorInfoCl *out_info,
//                              global float *out_data,
//                              global TensorInfoCl *in_info,
//                              global float *in_data,
//                              {{IndexType}} reductionStride,
//                              {{IndexType}} reductionSize,
//                              {{IndexType}} totalSlices,
//                              float init) {


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

  std::string uniqueName = "THClTensor_reduceContigDim_" + easycl::toString(ADims) + "_" + easycl::toString(BDims) + "_" +
    TypeParseTraits<IndexType>::name + "_" + modifyOp->operator2() + "_" + reduceOp->operator3();
  CLKernel *kernel = kernelBuilder.buildKernel(uniqueName, "THClReduce.cl", THClReduce_getKernelSource(), "THClTensor_reduceContigDim");
  // calculate workgroup sizes and stuff
  dim3 global_ws;
  for( int i = 0; i < 3; i++ ) {
      global_ws.vec[i] = grid.vec[i] * block.vec[i];
  }

  // set up tensorinfos
  TensorInfoCl outCl(out);
  TensorInfoCl inCl(in);

  if( !out.wrapper->isOnDevice() ) {
    out.wrapper->createOnDevice();
  }

  kernel->in(1, &outCl);
  kernel->out( out.wrapper );
  kernel->in(1, &inCl);
  kernel->in( in.wrapper );
  kernel->in((int)reductionSize);
  kernel->in((int)totalSlices);
  kernel->in(init);
  kernel->localFloats(smemSize / sizeof(float));

//  std::cout << "reductionsize " << reductionSize << std::endl;
//  std::cout << "totalSlices " << totalSlices << std::endl;
//  std::cout << "grid " << grid << std::endl;
//  std::cout << "block " << block << std::endl;
//  std::cout << "smemSize " << smemSize << std::endl;

  kernel->run(3, global_ws.vec, block.vec);
  THClState_getCl(state)->finish();

}
//THClTensor_reduceContigDim(global TensorInfoCl *out_info,
//                           global float *out_data,
//                           global TensorInfoCl *in_info,
//                           global float *in_data,
//                           int reductionSize,
//                           int totalSlices,
//                           float init,
//                           local float *smem) {

bool THClTensor_reduceDim(THClState* state,
                          THClTensor* out,
                          THClTensor* in,
                          float init,
                          const HasOperator2 *modifyOp,
                          const HasOperator3 *reduceOp,
                          int dim);

#undef THCL_NONCONTIG_REDUCE_BLOCK_SIZE

#endif // THCL_REDUCE_INC

