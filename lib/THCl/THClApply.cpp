#include "THClApply.h"

// Implementation of copyIgnoringOverlaps, defined after pointwiseApply2.
void THClTensor_copyIgnoringOverlaps(THClState* state,
                                       THClTensor* dst,
                                       THClTensor* src) {
  THClTensor_pointwiseApply2(state, dst, src, CopyOp<float>(),
                               ReadOnly, // ignore overwrites
                               ReadOnly);
}
