#include "THClStorageCopy.h"
#include "THClGeneral.h"

void THClStorage_copyFloat(THClState *state, THClStorage *self, struct THFloatStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
 // THClCheck(clMemcpy(self->data, src->data, self->size * sizeof(float), clMemcpyHostToDevice));
}

#define TH_CL_STORAGE_IMPLEMENT_COPY(TYPEC)                           \
  void THClStorage_copy##TYPEC(THClState *state, THClStorage *self, struct TH##TYPEC##Storage *src) \
  {                                                                     \
    THFloatStorage *buffer;                                             \
    THArgCheck(self->size == src->size, 2, "size does not match");      \
    buffer = THFloatStorage_newWithSize(src->size);                     \
    THFloatStorage_copy##TYPEC(buffer, src);                            \
    THClStorage_copyFloat(state, self, buffer);                              \
    THFloatStorage_free(buffer);                                        \
  }

TH_CL_STORAGE_IMPLEMENT_COPY(Byte)
TH_CL_STORAGE_IMPLEMENT_COPY(Char)
TH_CL_STORAGE_IMPLEMENT_COPY(Short)
TH_CL_STORAGE_IMPLEMENT_COPY(Int)
TH_CL_STORAGE_IMPLEMENT_COPY(Long)
TH_CL_STORAGE_IMPLEMENT_COPY(Double)

void THFloatStorage_copyCl(THClState *state, THFloatStorage *self, struct THClStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
//  THClCheck(clMemcpy(self->data, src->data, self->size * sizeof(float), clMemcpyDeviceToHost));
}

#define TH_CL_STORAGE_IMPLEMENT_COPYTO(TYPEC)                           \
  void TH##TYPEC##Storage_copyCl(THClState *state, TH##TYPEC##Storage *self, struct THClStorage *src) \
  {                                                                     \
    THFloatStorage *buffer;                                             \
    THArgCheck(self->size == src->size, 2, "size does not match");      \
    buffer = THFloatStorage_newWithSize(src->size);                     \
    THFloatStorage_copyCl(state, buffer, src);                               \
    TH##TYPEC##Storage_copyFloat(self, buffer);                         \
    THFloatStorage_free(buffer);                                        \
  }

TH_CL_STORAGE_IMPLEMENT_COPYTO(Byte)
TH_CL_STORAGE_IMPLEMENT_COPYTO(Char)
TH_CL_STORAGE_IMPLEMENT_COPYTO(Short)
TH_CL_STORAGE_IMPLEMENT_COPYTO(Int)
TH_CL_STORAGE_IMPLEMENT_COPYTO(Long)
TH_CL_STORAGE_IMPLEMENT_COPYTO(Double)
