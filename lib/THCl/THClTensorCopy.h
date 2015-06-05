#ifndef TH_CL_TENSOR_COPY_INC
#define TH_CL_TENSOR_COPY_INC

#include "THClTensor.h"
#include "THClGeneral.h"

THCL_API void THClTensor_copy(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_copyByte(THClState *state, THClTensor *self, THByteTensor *src);
THCL_API void THClTensor_copyChar(THClState *state, THClTensor *self, THCharTensor *src);
THCL_API void THClTensor_copyShort(THClState *state, THClTensor *self, THShortTensor *src);
THCL_API void THClTensor_copyInt(THClState *state, THClTensor *self, THIntTensor *src);
THCL_API void THClTensor_copyLong(THClState *state, THClTensor *self, THLongTensor *src);
THCL_API void THClTensor_copyFloat(THClState *state, THClTensor *self, THFloatTensor *src);
THCL_API void THClTensor_copyDouble(THClState *state, THClTensor *self, THDoubleTensor *src);

THCL_API void THByteTensor_copyCl(THClState *state, THByteTensor *self, THClTensor *src);
THCL_API void THCharTensor_copyCl(THClState *state, THCharTensor *self, THClTensor *src);
THCL_API void THShortTensor_copyCl(THClState *state, THShortTensor *self, THClTensor *src);
THCL_API void THIntTensor_copyCl(THClState *state, THIntTensor *self, THClTensor *src);
THCL_API void THLongTensor_copyCl(THClState *state, THLongTensor *self, THClTensor *src);
THCL_API void THFloatTensor_copyCl(THClState *state, THFloatTensor *self, THClTensor *src);
THCL_API void THDoubleTensor_copyCl(THClState *state, THDoubleTensor *self, THClTensor *src);
THCL_API void THClTensor_copyCl(THClState *state, THClTensor *self, THClTensor *src);

#endif
