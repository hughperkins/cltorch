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
bool THClTensor_reduceDim(THClState* state,
                          THClTensor* out,
                          THClTensor* in,
                          float init,
                          const HasOperator2 *modifyOp,
                          const HasOperator3 *reduceOp,
                          int dim);

#undef THCL_NONCONTIG_REDUCE_BLOCK_SIZE

#endif // THCL_REDUCE_INC

