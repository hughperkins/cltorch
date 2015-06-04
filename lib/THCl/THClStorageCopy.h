#ifndef THCL_STORAGE_COPY_INC
#define THCL_STORAGE_COPY_INC

#include "THClStorage.h"
#include "THClGeneral.h"

/* Support for copy between different Storage types */

THCL_API void THClStorage_rawCopy(THClState *state, THClStorage *storage, float *src);
THCL_API void THClStorage_copy(THClState *state, THClStorage *storage, THClStorage *src);
THCL_API void THClStorage_copyByte(THClState *state, THClStorage *storage, struct THByteStorage *src);
THCL_API void THClStorage_copyChar(THClState *state, THClStorage *storage, struct THCharStorage *src);
THCL_API void THClStorage_copyShort(THClState *state, THClStorage *storage, struct THShortStorage *src);
THCL_API void THClStorage_copyInt(THClState *state, THClStorage *storage, struct THIntStorage *src);
THCL_API void THClStorage_copyLong(THClState *state, THClStorage *storage, struct THLongStorage *src);
THCL_API void THClStorage_copyFloat(THClState *state, THClStorage *storage, struct THFloatStorage *src);
THCL_API void THClStorage_copyDouble(THClState *state, THClStorage *storage, struct THDoubleStorage *src);

THCL_API void THByteStorage_copyCl(THClState *state, THByteStorage *self, struct THClStorage *src);
THCL_API void THCharStorage_copyCl(THClState *state, THCharStorage *self, struct THClStorage *src);
THCL_API void THShortStorage_copyCl(THClState *state, THShortStorage *self, struct THClStorage *src);
THCL_API void THIntStorage_copyCl(THClState *state, THIntStorage *self, struct THClStorage *src);
THCL_API void THLongStorage_copyCl(THClState *state, THLongStorage *self, struct THClStorage *src);
THCL_API void THFloatStorage_copyCl(THClState *state, THFloatStorage *self, struct THClStorage *src);
THCL_API void THDoubleStorage_copyCl(THClState *state, THDoubleStorage *self, struct THClStorage *src);
THCL_API void THClStorage_copyCl(THClState *state, THClStorage *self, THClStorage *src);

#endif
