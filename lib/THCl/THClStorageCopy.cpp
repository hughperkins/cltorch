#include "THClStorageCopy.h"
#include "THClGeneral.h"

void THClStorage_rawCopy(THClState *state, THClStorage *self, float *src)
{
//  THClCheck(clMemcpyAsync(self->data, src, self->size * sizeof(float), clMemcpyDeviceToDevice, THClState_getCurrentStream(state)));
}

void THClStorage_copy(THClState *state, THClStorage *self, THClStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
//  THClCheck(clMemcpyAsync(self->data, src->data, self->size * sizeof(float), clMemcpyDeviceToDevice, THClState_getCurrentStream(state)));
}

void THClStorage_copyCl(THClState *state, THClStorage *self, THClStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
//  THClCheck(clMemcpyAsync(self->data, src->data, self->size * sizeof(float), clMemcpyDeviceToDevice, THClState_getCurrentStream(state)));
}
