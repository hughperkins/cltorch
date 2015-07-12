// from lib/THC/THCReduceAll.cuh:

#pragma once

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

//long getReduceAllBlockSize(THClState *state);
//long getTwoPassReductionSize(THClState *state);

bool THClTensor_reduceAll(THClState* state,
                            THClTensor* in,
                            const HasOperator2 *modifyOp,
                            const HasOperator3 *reduceOp,
                            float init,
                            float* p_result);

//bool isTwoPassReductionSize(THClState *state, long elements);
//long getTwoPassBlocks(THClState* state, long elements);

//void getPass1ReduceBlockGrid(THClState* state, long elements,
//                                    dim3& grid, dim3& block);
//void getPass2ReduceBlockGrid(THClState* state, long elements,
//                                    dim3& grid, dim3& block);
//void getSinglePassReduceBlockGrid(THClState *state, long elements,
//                                         dim3& grid, dim3& block);

