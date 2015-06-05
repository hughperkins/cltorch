#include "THClTensorCopy.h"
#include "THClGeneral.h"
#include "THClTensor.h"

/* specific methods */

void THClTensor_copyFloat(THClState *state, THClTensor *self, struct THFloatTensor *src)
{
  THError("Not implemented");
//  THArgCheck(THClTensor_nElement(state, self) == THFloatTensor_nElement(src), 2, "sizes do not match"); 

//  {
//    THClTensor *selfc = THClTensor_newContiguous(state, self);
//    src = THFloatTensor_newContiguous(src);
//  
//    THClCheck(cudaMemcpy(selfc->storage->data + selfc->storageOffset, src->storage->data + src->storageOffset, THFloatTensor_nElement(src) * sizeof(float), cudaMemcpyHostToDevice));

//    THFloatTensor_free(src);
//    THClTensor_freeCopyTo(state, selfc, self);
//  }
}

/* everything comes down to copy to a tensor of floats */
#define IMPLEMENT_TH_CL_TENSOR_COPY(TYPEC)                            \
void THClTensor_copy##TYPEC(THClState *state, THClTensor *self, struct TH##TYPEC##Tensor *src) \
{                                                                       \
  THArgCheck(THClTensor_nElement(state, self) == TH##TYPEC##Tensor_nElement(src), 2, "sizes do not match"); \
                                                                        \
  {                                                                     \
    THLongStorage *size = TH##TYPEC##Tensor_newSizeOf(src);             \
    THFloatTensor *srcf = THFloatTensor_newWithSize(size, NULL);        \
                                                                        \
    THFloatTensor_copy##TYPEC(srcf, src);                               \
    THClTensor_copyFloat(state, self, srcf);                                 \
                                                                        \
    THLongStorage_free(size);                                           \
    THFloatTensor_free(srcf);                                           \
  }                                                                     \
}

IMPLEMENT_TH_CL_TENSOR_COPY(Byte)
IMPLEMENT_TH_CL_TENSOR_COPY(Char)
IMPLEMENT_TH_CL_TENSOR_COPY(Short)
IMPLEMENT_TH_CL_TENSOR_COPY(Int)
IMPLEMENT_TH_CL_TENSOR_COPY(Long)
IMPLEMENT_TH_CL_TENSOR_COPY(Double)

/* copyCl */

void THFloatTensor_copyCl(THClState *state, THFloatTensor *self, struct THClTensor *src)
{
  THError("Not implemented");
/*  THArgCheck(THFloatTensor_nElement(self) == THClTensor_nElement(state, src), 2, "sizes do not match");*/

/*  {*/
/*    THFloatTensor *selfc = THFloatTensor_newContiguous(self);*/
/*    src = THClTensor_newContiguous(state, src);*/

/*    THClCheck(cudaMemcpy(selfc->storage->data + selfc->storageOffset,*/
/*                           src->storage->data + src->storageOffset,*/
/*                           THClTensor_nElement(state, src) * sizeof(float),*/
/*                           cudaMemcpyDeviceToHost));*/

/*    THClTensor_free(state, src);*/
/*    THFloatTensor_freeCopyTo(selfc, self);*/
/*  }*/
}

#define IMPLEMENT_TH_CL_TENSOR_COPY_TO(TYPEC)                                                          \
  void TH##TYPEC##Tensor_copyCl(THClState *state, TH##TYPEC##Tensor *self, struct THClTensor *src) \
  {                                                                                                      \
    THArgCheck(TH##TYPEC##Tensor_nElement(self) == THClTensor_nElement(state, src), 2, "sizes do not match"); \
                                                                                                         \
    {                                                                                                    \
      THLongStorage *size = THClTensor_newSizeOf(state, src);                                          \
      THFloatTensor *srcf = THFloatTensor_newWithSize(size, NULL);                                       \
                                                                                                         \
      THFloatTensor_copyCl(state, srcf, src);                                                          \
      TH##TYPEC##Tensor_copyFloat(self, srcf);                                                           \
                                                                                                         \
      THLongStorage_free(size);                                                                          \
      THFloatTensor_free(srcf);                                                                          \
    }                                                                                                    \
  }

IMPLEMENT_TH_CL_TENSOR_COPY_TO(Byte)
IMPLEMENT_TH_CL_TENSOR_COPY_TO(Char)
IMPLEMENT_TH_CL_TENSOR_COPY_TO(Short)
IMPLEMENT_TH_CL_TENSOR_COPY_TO(Int)
IMPLEMENT_TH_CL_TENSOR_COPY_TO(Long)
IMPLEMENT_TH_CL_TENSOR_COPY_TO(Double)

void THClTensor_copyCl(THClState *state, THClTensor *self, THClTensor *src)
{
  THClTensor_copy(state, self, src);
}
