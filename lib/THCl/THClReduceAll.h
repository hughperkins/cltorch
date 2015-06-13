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
                            float* out,
                            int outOnDevice);

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

void kernelLaunch_THClTensor_reduceAllPass1(
                     THClState* state,
                     int ADims,
                     const TensorInfoCl& inCl,
                     CLWrapper *in_data,
                     long totalElements,
                     float init,
                     const HasOperator2 *modifyOp,
                     const HasOperator3 *reduceOp,
                     CLWrapper* scratch
    );

void kernelLaunch_THClTensor_reduceAllPass2(
                     THClState* state,
                     int numPass1Blocks,
                     float init,
                     const HasOperator3 *reduceOp,
                     CLWrapper *scratch,
                     CLWrapper* devOut
    );

void kernelLaunch_THClTensor_reduceAll(
                     THClState* state,
                     int ADims,
                     const TensorInfoCl& inCl,
                     CLWrapper *in_data,
                     long totalElements,
                     float init,
                     const HasOperator2 *modifyOp,
                     const HasOperator3 *reduceOp,
                     CLWrapper* devOut
    );

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

  TensorInfoCl inCl(in);
  if (isTwoPassReductionSize(totalElements)) {
    getPass1ReduceBlockGrid(state, totalElements, grid, block);
    size_t smemSize = block.x() * sizeof(float);

    kernelLaunch_THClTensor_reduceAllPass1(
        state,
        ADims,
          inCl, in.wrapper,
           (IndexType) totalElements, init, modifyOp, reduceOp,
        THClState_getCurrentDeviceScratchSpace(state));
//    THClTensor_reduceAllPass1<ModifyOp, ReduceOp, IndexType, ADims>
//      <<<grid, block, smemSize, THClState_getCurrentStream(state)>>>(
//        in, (IndexType) totalElements, init, modifyOp, reduceOp,
//        (float*) THClState_getCurrentDeviceScratchSpace(state));

    int numPass1Blocks = grid.x();
    getPass2ReduceBlockGrid(state, totalElements, grid, block);
    smemSize = block.x() * sizeof(float);

    kernelLaunch_THClTensor_reduceAllPass2(
        state,
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

    kernelLaunch_THClTensor_reduceAll(
        state,
        ADims, 
        inCl,
        in.wrapper,
        (IndexType) totalElements, init, modifyOp, reduceOp, devOut);
//    THClTensor_reduceAll<ModifyOp, ReduceOp, IndexType, ADims>
//      <<<grid, block, smemSize, THClState_getCurrentStream(state)>>>(
//        in, (IndexType) totalElements, init, modifyOp, reduceOp, devOut);
  }
}


// #undef THCL_REDUCE_ALL_BLOCK_SIZE
// #undef THCL_TWO_PASS_REDUCTION_SIZE

#endif // THCL_REDUCEALL_INC

