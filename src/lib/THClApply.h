#ifndef THCL_APPLY_INC
#define THCL_APPLY_INC

#include "THClGeneral.h"
#include "THClTensor.h"
#include "THClOperators.h"
#include "THClReduceApplyUtils.h"

//
// This file contains pointwise operation functions and kernels that
// work on both contiguous and non-contiguous tensor arguments of
// arbitrary (up to MAX_CLTORCH_DIMS) dimensioned arguments without
// copying or temporary storage.
//

int getWorkgroupSize(THClState *state, int device);
dim3 getApplyBlock(THClState *state, int device);
dim3 getApplyGrid(THClState* state, int device, long totalElements);

bool THClTensor_pointwiseApply1(THClState* state,
                                  THClTensor* a,
                                  const HasOperator1 *op,
                                  TensorArgType aType = ReadWrite);
bool THClTensor_pointwiseApply2(THClState* state,
                                  THClTensor* a,
                                  THClTensor* b,
                                  const HasOperator2 *op,
                                  TensorArgType aType = ReadWrite,
                                  TensorArgType bType = ReadOnly);
bool THClTensor_pointwiseApply3(THClState* state,
                                  THClTensor* a,
                                  THClTensor* b,
                                  THClTensor* c,
                                  const HasOperator3 *op,
                                  TensorArgType aType = ReadWrite,
                                  TensorArgType bType = ReadOnly,
                                  TensorArgType cType = ReadOnly);

#endif // THCL_APPLY_INC

