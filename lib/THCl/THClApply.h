#pragma once

#include "THClTensorCopy.h"
#include "THClReduceApplyUtils.h"
#include "templates/TemplatedKernel.h"
#include "util/easycl_stringhelper.h"
#include "EasyCL.h"
#include "CLKernel_structs.h"
#include "THClTypeParseTraits.h"
#include "THClKernels.h"
#include "DeviceInfo.h"
#include "util/StatefulTimer.h"

#include <string>

//
// This file contains pointwise operation functions and kernels that
// work on both contiguous and non-contiguous tensor arguments of
// arbitrary (up to MAX_CLTORCH_DIMS) dimensioned arguments without
// copying or temporary storage.
//

std::string getApplyDv2_template();

// Called when we are copying into an overlapping index `dst`, but
// we don't care which writer wins. Hacky but it works.
void THClTensor_copyIgnoringOverlaps(THClState* state,
                                       THClTensor* dst,
                                       THClTensor* src);
int getWorkgroupSize(THClState *state);
dim3 getApplyBlock(THClState *state);
bool getApplyGrid(THClState* state, long totalElements, dim3& grid);

template< typename IndexType >
void kernelLaunch_pointwiseApply1( THClState *state, dim3 grid, dim3 block, int A, TensorInfo<IndexType> aInfo, IndexType totalElements, HasOperator1 const * op );
template< typename IndexType >
void kernelLaunch_pointwiseApply2( THClState *state, dim3 grid, dim3 block, int A, int B, TensorInfo<IndexType> aInfo, TensorInfo<IndexType> bInfo, IndexType totalElements, HasOperator2 const*op );
template< typename IndexType >
void kernelLaunch_pointwiseApply3( THClState *state, dim3 grid, dim3 block, int A, int B, int C, TensorInfo<IndexType> aInfo, TensorInfo<IndexType> bInfo, TensorInfo<IndexType> cInfo, IndexType totalElements, HasOperator3 const*op );

bool THClTensor_pointwiseApply1(THClState* state,
                                  THClTensor* a,
                                  HasOperator1 const*op,
                                  TensorArgType aType = ReadWrite);
bool THClTensor_pointwiseApply2(THClState* state,
                                  THClTensor* a,
                                  THClTensor* b,
                                  HasOperator2 const*op,
                                  TensorArgType aType = ReadWrite,
                                  TensorArgType bType = ReadOnly);
bool THClTensor_pointwiseApply3(THClState* state,
                                  THClTensor* a,
                                  THClTensor* b,
                                  THClTensor* c,
                                  HasOperator3 const*op,
                                  TensorArgType aType = ReadWrite,
                                  TensorArgType bType = ReadOnly,
                                  TensorArgType cType = ReadOnly);

