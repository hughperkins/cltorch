// from lib/THC/THCReduceAll.cuh:

#ifndef THCL_REDUCEALL_INC
#define THCL_REDUCEALL_INC

//
// This file contains dimension reduction operation functions and
// kernels that work on both contiguous and non-contiguous tensor
// arguments of arbitrary (up to MAX_CLTORCH_DIMS) dimensioned
// arguments without copying or temporary storage, for reducing an
// entire tensor to one value.
//

#include "THClReduceApplyUtils.h"
#include "THClDeviceUtils.h"
#include "templates/TemplatedKernel.h"
#include "THClTypeParseTraits.h"
#include "THClKernels.h"
#include "util/StatefulTimer.h"

// Size per each reduction block
//#define THCL_REDUCE_ALL_BLOCK_SIZE 1024L

inline long getReduceAllBlockSize(THClState *state) {
  int blockSize = 1024;
  int maxWorkgroupSize = ((easycl::DeviceInfo *)state->deviceInfoByDevice[state->currentDevice])->maxWorkGroupSize;
  if( blockSize > maxWorkgroupSize ) {
    blockSize = maxWorkgroupSize;
  }
  return blockSize;
}

// Cutoff size for two-pass reduction
// #define THCL_TWO_PASS_REDUCTION_SIZE 2048L
// I wonder if this is a function of the block size above?  logically it 
// probably is...

inline long getTwoPassReductionSize(THClState *state) {
  return getReduceAllBlockSize(state) * 2;
}

std::string THClReduceAll_getKernelTemplate();
bool THClTensor_reduceAll(THClState* state,
                            THClTensor* in,
                            const HasOperator2 *modifyOp,
                            const HasOperator3 *reduceOp,
                            float init,
                            float* p_result);

// Perform a two-pass reduction if the tensor is large enough to
// warrant it.
inline bool isTwoPassReductionSize(THClState *state, long elements) {
  return (elements > getTwoPassReductionSize(state));
}

inline long getTwoPassBlocks(THClState* state, long elements) {
  long numBlocks = THClCeilDiv(elements, getReduceAllBlockSize(state));

  // We can only have as many blocks as there is scratch space
  size_t scratchSpace =
    THClState_getCurrentDeviceScratchSpaceSize(state) / sizeof(float);
  THAssert(scratchSpace > 0);

  if (numBlocks > (long)scratchSpace) {
    numBlocks = scratchSpace;
  }

  return numBlocks;
}

// Get the block/grid size that we want
inline void getPass1ReduceBlockGrid(THClState* state, long elements,
                                    dim3& grid, dim3& block) {
  grid = dim3(getTwoPassBlocks(state, elements));
  block = dim3(getReduceAllBlockSize(state));
}

inline void getPass2ReduceBlockGrid(THClState* state, long elements,
                                    dim3& grid, dim3& block) {
  grid = dim3(1);
  // We only need as many threads as there were blocks originally
  block = dim3(getTwoPassBlocks(state, elements));
}

inline void getSinglePassReduceBlockGrid(THClState *state, long elements,
                                         dim3& grid, dim3& block) {
  grid = dim3(1);
  block = dim3(getReduceAllBlockSize(state));
}

template< typename IndexType >
void kernelLaunch_THClTensor_reduceAllPass1(
                     THClState* state,
                     dim3 &grid, dim3 &block, size_t smemSize,
                     int ADims,
                     const TensorInfo<IndexType> & in,
//                     CLWrapper *in_data,
                     long totalElements,
                     float init,
                     const HasOperator2 *modifyOp,
                     const HasOperator3 *reduceOp,
                     CLWrapper* scratch
    ){
  StatefulTimer::timeCheck("ReduceAllPass1 START");
  std::string uniqueName = "THClTensor_reduceAllPass1_" + easycl::toString(ADims) + "_" + modifyOp->operator2() + "_" + reduceOp->operator3();
  EasyCL *cl = THClState_getCl(state);
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("ReduceAllPass1 1aa");
  } else {
    std::vector< int > dims;
    if( ADims >= 0 ) {
      dims.push_back(ADims);
    }
    TemplatedKernel kernelBuilder(THClState_getCl(state));
    kernelBuilder
      .set("include_THClDeviceUtils", THClDeviceUtils_getKernelTemplate())
      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
      .set("dims", dims)
      .set("dim1", ADims)
      .set("modify_operation", modifyOp->operator2())
      .set("reduce_operation", reduceOp->operator3())
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
      .set("IndexType", TypeParseTraits<IndexType>::name)
    ;

    kernel = kernelBuilder.buildKernel( uniqueName, "THClReduceAll.cl", THClReduceAll_getKernelTemplate(), "THClTensor_reduceAllPass1" );
  }

  THClKernels k(state, kernel);
  k.in(in);
  k.in((int)totalElements);
  k.in(init);
  k.out(scratch);
  k.localFloats(smemSize / sizeof(float));
  k.run(grid, block);
  StatefulTimer::timeCheck("ReduceAllPass1 END");
}

template< typename IndexType >
void kernelLaunch_THClTensor_reduceAllPass2(
                     THClState* state,
                     dim3 &grid, dim3 &block, size_t smemSize,
                     int numPass1Blocks,
                     float init,
                     const HasOperator3 *reduceOp,
                     CLWrapper *scratch,
                     CLWrapper* devOut
    ){
  StatefulTimer::timeCheck("ReduceAllPass2 START");
  std::string uniqueName = "THClTensor_reduceAllPass2_" + reduceOp->operator3();
  EasyCL *cl = THClState_getCl(state);
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("ReduceAllPass2 1aa");
  } else {
    TemplatedKernel kernelBuilder(THClState_getCl(state));
    std::vector< int > dims;
    kernelBuilder
      .set("include_THClDeviceUtils", THClDeviceUtils_getKernelTemplate())
      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
      .set("dims", dims)
      .set("dim1", -2)
  //    .set("modify_operation", modifyOp->operator2())
      .set("reduce_operation", reduceOp->operator3())
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
      .set("IndexType", TypeParseTraits<IndexType>::name)
    ;

    kernel = kernelBuilder.buildKernel( uniqueName, "THClReduceAll.cl", THClReduceAll_getKernelTemplate(), "THClTensor_reduceAllPass2" );
  }

  THClKernels k(state, kernel);
  k.in(numPass1Blocks);
  k.in(init);
  k.in(scratch);
  k.out(devOut);
  k.localFloats(smemSize / sizeof(float));
  k.run(grid, block);

  StatefulTimer::timeCheck("ReduceAllPass2 End");
}

template< typename IndexType >
void kernelLaunch_THClTensor_reduceAll(
                     THClState* state,
                     dim3 &grid, dim3 &block, size_t smemSize,
                     int ADims,
                     const TensorInfo<IndexType> &in,
//                     CLWrapper *in_data,
                     long totalElements,
                     float init,
                     const HasOperator2 *modifyOp,
                     const HasOperator3 *reduceOp,
                     CLWrapper* devOut
    ){
  StatefulTimer::timeCheck("ReduceAll START");
  std::string uniqueName = "THClTensor_reduceAll_" + easycl::toString(ADims) + "_" + modifyOp->operator2() + "_" + reduceOp->operator3();
  EasyCL *cl = THClState_getCl(state);
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("ReduceAllPass2 1aa");
  } else {
    std::vector< int > dims;
    if( ADims >= 0 ) {
      dims.push_back(ADims);
    }
    TemplatedKernel kernelBuilder(THClState_getCl(state));
    kernelBuilder
      .set("include_THClDeviceUtils", THClDeviceUtils_getKernelTemplate())
      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
      .set("dims", dims)
      .set("dim1", ADims)
      .set("modify_operation", modifyOp->operator2())
      .set("reduce_operation", reduceOp->operator3())
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
      .set("IndexType", TypeParseTraits<IndexType>::name)
    ;

    kernel = kernelBuilder.buildKernel( uniqueName, "THClReduceAll.cl", THClReduceAll_getKernelTemplate(), "THClTensor_reduceAll" );
  }

  THClKernels k(state, kernel);
  k.in(in);
  k.in((int)totalElements);
  k.in(init);
  k.out(devOut);
  k.localFloats(smemSize / sizeof(float));
  k.run(grid, block);
  StatefulTimer::timeCheck("ReduceAll END");
}

template <typename IndexType>
void callReduceAll(THClState* state,
                   int ADims,
                   const TensorInfo<IndexType>& in,
                   long totalElements,
                   float init,
                   const HasOperator2 *modifyOp,
                   const HasOperator3 *reduceOp,
                   CLWrapper* devOut) {
  dim3 grid;
  dim3 block;

//  TensorInfoCl inCl(in);
  if (isTwoPassReductionSize(state, totalElements)) {
    getPass1ReduceBlockGrid(state, totalElements, grid, block);
    size_t smemSize = block.x() * sizeof(float);

    kernelLaunch_THClTensor_reduceAllPass1<IndexType>(
        state,
        grid, block, smemSize,
        ADims,
          in,
           (IndexType) totalElements, init, modifyOp, reduceOp,
        THClState_getCurrentDeviceScratchSpace(state)->wrapper);
//    THClTensor_reduceAllPass1<ModifyOp, ReduceOp, IndexType, ADims>
//      <<<grid, block, smemSize, THClState_getCurrentStream(state)>>>(
//        in, (IndexType) totalElements, init, modifyOp, reduceOp,
//        (float*) THClState_getCurrentDeviceScratchSpace(state));

    int numPass1Blocks = grid.x();
    getPass2ReduceBlockGrid(state, totalElements, grid, block);
    smemSize = block.x() * sizeof(float);

//    THError("not implemented");
    kernelLaunch_THClTensor_reduceAllPass2<IndexType>(
        state,
        grid, block, smemSize,
        numPass1Blocks, init, reduceOp,
        THClState_getCurrentDeviceScratchSpace(state)->wrapper,
        devOut);
//    THClTensor_reduceAllPass2<ReduceOp, IndexType>
//      <<<grid, block, smemSize, THClState_getCurrentStream(state)>>>(
//        numPass1Blocks, init, reduceOp,
//        (float*) THClState_getCurrentDeviceScratchSpace(state),
//        devOut);
//    THError("Not implemented");
  } else {
    getSinglePassReduceBlockGrid(state, totalElements, grid, block);
    size_t smemSize = block.x() * sizeof(float);

//    THError("Not implemented");
    kernelLaunch_THClTensor_reduceAll<IndexType>(
        state,
        grid, block, smemSize,
        ADims, 
        in,
//        in.wrapper,
        (IndexType) totalElements, init, modifyOp, reduceOp, devOut);
//    THClTensor_reduceAll<ModifyOp, ReduceOp, IndexType, ADims>
//      <<<grid, block, smemSize, THClState_getCurrentStream(state)>>>(
//        in, (IndexType) totalElements, init, modifyOp, reduceOp, devOut);
  }
}


// #undef THCL_REDUCE_ALL_BLOCK_SIZE
// #undef THCL_TWO_PASS_REDUCTION_SIZE

#endif // THCL_REDUCEALL_INC

