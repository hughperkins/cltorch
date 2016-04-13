#pragma once

// from lib/THC/THCTensorSort.h:

#include "THClTensor.h"

/* Performs an in-place sort of (keys, values). Only works for slice sizes
   <= 2048 at the moment (slice size == size of keys/values dim `dim`) */
THCL_API void THClTensor_sortKeyValueInplace(THClState* state,
                                              THClTensor* keys,
                                              THClTensor* values,
                                              int dim, int order);

/* Performs an out-of-place sort of `input`, returning the per-slice indices
   in `indices` and the sorted values in `sorted` */
THCL_API void THClTensor_sort(THClState* state,
                               THClTensor* sorted,
                               THClTensor* indices,
                               THClTensor* input,
                               int dim, int order);

