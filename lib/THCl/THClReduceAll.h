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

// Size per each reduction block
#define THCL_REDUCE_ALL_BLOCK_SIZE 1024L

// Cutoff size for two-pass reduction
#define THCL_TWO_PASS_REDUCTION_SIZE 2048L

std::string THClReduceAll_getKernelTemplate();
bool THClTensor_reduceAll(THClState* state,
                            THClTensor* in,
                            const HasOperator2 *modifyOp,
                            const HasOperator3 *reduceOp,
                            float init,
                            float* p_result);

// Perform a two-pass reduction if the tensor is large enough to
// warrant it.
inline bool isTwoPassReductionSize(long elements) {
  return (elements > THCL_TWO_PASS_REDUCTION_SIZE);
}

inline long getTwoPassBlocks(THClState* state, long elements) {
  long numBlocks = THClCeilDiv(elements, THCL_REDUCE_ALL_BLOCK_SIZE);

  // We can only have as many blocks as there is scratch space
  size_t scratchSpace =
    THClState_getCurrentDeviceScratchSpaceSize(state) / sizeof(float);
  THAssert(scratchSpace > 0);

  if (numBlocks > scratchSpace) {
    numBlocks = scratchSpace;
  }

  return numBlocks;
}

// Get the block/grid size that we want
inline void getPass1ReduceBlockGrid(THClState* state, long elements,
                                    dim3& grid, dim3& block) {
  grid = dim3(getTwoPassBlocks(state, elements));
  block = dim3(THCL_REDUCE_ALL_BLOCK_SIZE);
}

inline void getPass2ReduceBlockGrid(THClState* state, long elements,
                                    dim3& grid, dim3& block) {
  grid = dim3(1);
  // We only need as many threads as there were blocks originally
  block = dim3(getTwoPassBlocks(state, elements));
}

inline void getSinglePassReduceBlockGrid(long elements,
                                         dim3& grid, dim3& block) {
  grid = dim3(1);
  block = dim3(THCL_REDUCE_ALL_BLOCK_SIZE);
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
  EasyCL *cl = THClState_getCl(state);
  TemplatedKernel kernelBuilder(cl);
  kernelBuilder
      .set("include_THClDeviceutils", THClDeviceUtils_getKernelTemplate())
  ;

  THError("Not implemented");
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
  TemplatedKernel kernelBuilder(THClState_getCl(state));
  kernelBuilder
      .set("include_THClDeviceutils", THClDeviceUtils_getKernelTemplate())
  ;

  THError("Not implemented");
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

  std::string uniqueName = "THClTensor_reduceAll_" + easycl::toString(ADims) + "_" + modifyOp->operator2() + "_" + reduceOp->operator3();
  CLKernel *kernel = kernelBuilder.buildKernel( uniqueName, "THClReduceAll.cl", THClReduceAll_getKernelTemplate(), "THClTensor_reduceAll" );
  // calculate workgroup sizes and stuff
  dim3 global_ws;
  for( int i = 0; i < 3; i++ ) {
      global_ws.vec[i] = grid.vec[i] * block.vec[i];
  }

  // set up tensorinfos
  TensorInfoCl inCl(in);

  if( !in.wrapper->isOnDevice() ) {
    in.wrapper->createOnDevice();
  }

  kernel->in(1, &inCl);
  kernel->inout( in.wrapper );

  if( totalElements > ( 1l << 30 )) {
    throw std::runtime_error("Error: out of bounds for totalelements=" + easycl::toString(totalElements));
  }
  kernel->in( (int)totalElements );
  kernel->run(3, global_ws.vec, block.vec);
  THClState_getCl(state)->finish();

  THError("Not implemented");
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
  if (isTwoPassReductionSize(totalElements)) {
    getPass1ReduceBlockGrid(state, totalElements, grid, block);
    size_t smemSize = block.x() * sizeof(float);

    kernelLaunch_THClTensor_reduceAllPass1<IndexType>(
        state,
        grid, block, smemSize,
        ADims,
          in, /* in.wrapper,*/
           (IndexType) totalElements, init, modifyOp, reduceOp,
        THClState_getCurrentDeviceScratchSpace(state));
//    THClTensor_reduceAllPass1<ModifyOp, ReduceOp, IndexType, ADims>
//      <<<grid, block, smemSize, THClState_getCurrentStream(state)>>>(
//        in, (IndexType) totalElements, init, modifyOp, reduceOp,
//        (float*) THClState_getCurrentDeviceScratchSpace(state));

    int numPass1Blocks = grid.x();
    getPass2ReduceBlockGrid(state, totalElements, grid, block);
    smemSize = block.x() * sizeof(float);

    kernelLaunch_THClTensor_reduceAllPass2<IndexType>(
        state,
        grid, block, smemSize,
        numPass1Blocks, init, reduceOp,
        THClState_getCurrentDeviceScratchSpace(state),
        devOut);
//    THClTensor_reduceAllPass2<ReduceOp, IndexType>
//      <<<grid, block, smemSize, THClState_getCurrentStream(state)>>>(
//        numPass1Blocks, init, reduceOp,
//        (float*) THClState_getCurrentDeviceScratchSpace(state),
//        devOut);

  } else {
    getSinglePassReduceBlockGrid(totalElements, grid, block);
    size_t smemSize = block.x() * sizeof(float);

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

