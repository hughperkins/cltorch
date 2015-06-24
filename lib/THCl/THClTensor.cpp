//extern "C" {
    #include "THClGeneral.h"
    #include "THClTensor.h"
    #include "THClTensorCopy.h"
    #include "THAtomic.h"
//}
#include "util/easycl_stringhelper.h"

#include <iostream>

using namespace std;

/**** access methods ****/
THClStorage *THClTensor_storage(THClState *state, const THClTensor *self)
{
  return self->storage;
}

long THClTensor_storageOffset(THClState *state, const THClTensor *self)
{
  return self->storageOffset;
}

int THClTensor_nDimension(THClState *state, const THClTensor *self)
{
  return self->nDimension;
}

long THClTensor_size(THClState *state, const THClTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->size[dim];
}

long THClTensor_stride(THClState *state, const THClTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->stride[dim];
}

THLongStorage *THClTensor_newSizeOf(THClState *state, THClTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THClTensor_newStrideOf(THClState *state, THClTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

float *THClTensor_data(THClState *state, const THClTensor *self)
{
  if(self->storage)
    return (self->storage->data+self->storageOffset);
  else
    return NULL;
}
CLWrapper *THClTensor_wrapper(THClState *state, const THClTensor *self)
{
  if(self->storage)
    return self->storage->wrapper;
  else
    return NULL;
}

void THClTensor_setFlag(THClState *state, THClTensor *self, const char flag)
{
  self->flag |= flag;
}

void THClTensor_clearFlag(THClState *state, THClTensor *self, const char flag)
{
  self->flag &= ~flag;
}

/**** creation methods ****/

static void THClTensor_rawInit(THClState *state, THClTensor *self);
static void THClTensor_rawSet(THClState *state, THClTensor *self, THClStorage *storage, long storageOffset, int nDimension, long *size, long *stride);
static void THClTensor_rawResize(THClState *state, THClTensor *self, int nDimension, long *size, long *stride);


/* Empty init */
THClTensor *THClTensor_new(THClState *state)
{
  THClTensor *self = (THClTensor*)THAlloc(sizeof(THClTensor));
  THClTensor_rawInit(state, self);
  return self;
}

/* Pointer-copy init */
THClTensor *THClTensor_newWithTensor(THClState *state, THClTensor *tensor)
{
  THClTensor *self = (THClTensor*)THAlloc(sizeof(THClTensor));
  THClTensor_rawInit(state, self);
  THClTensor_rawSet(state,
                      self,
                      tensor->storage,
                      tensor->storageOffset,
                      tensor->nDimension,
                      tensor->size,
                      tensor->stride);
  return self;
}

/* Storage init */
THClTensor *THClTensor_newWithStorage(THClState *state, THClStorage *storage, long storageOffset, THLongStorage *size, THLongStorage *stride)
{
  THClTensor *self = (THClTensor*)THAlloc(sizeof(THClTensor));
  if(size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");

  THClTensor_rawInit(state, self);
  THClTensor_rawSet(state,
                      self,
                      storage,
                      storageOffset,
                      (size ? size->size : (stride ? stride->size : 0)),
                      (size ? size->data : NULL),
                      (stride ? stride->data : NULL));

  return self;
}
THClTensor *THClTensor_newWithStorage1d(THClState *state, THClStorage *storage, long storageOffset,
                               long size0, long stride0)
{
  return THClTensor_newWithStorage4d(state, storage, storageOffset, size0, stride0, -1, -1,  -1, -1,  -1, -1);
}

THClTensor *THClTensor_newWithStorage2d(THClState *state, THClStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1)
{
  return THClTensor_newWithStorage4d(state, storage, storageOffset, size0, stride0, size1, stride1,  -1, -1,  -1, -1);
}

THClTensor *THClTensor_newWithStorage3d(THClState *state, THClStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2)
{
  return THClTensor_newWithStorage4d(state, storage, storageOffset, size0, stride0, size1, stride1,  size2, stride2,  -1, -1);
}

THClTensor *THClTensor_newWithStorage4d(THClState *state, THClStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2,
                               long size3, long stride3)
{
  long size[4] = {size0, size1, size2, size3};
  long stride[4] = {stride0, stride1, stride2, stride3};

  THClTensor *self = (THClTensor*)THAlloc(sizeof(THClTensor));
  THClTensor_rawInit(state, self);
  THClTensor_rawSet(state, self, storage, storageOffset, 4, size, stride);

  return self;
}

THClTensor *THClTensor_newWithSize(THClState *state, THLongStorage *size, THLongStorage *stride)
{
  return THClTensor_newWithStorage(state, NULL, 0, size, stride);
}

THClTensor *THClTensor_newWithSize1d(THClState *state, long size0)
{
  return THClTensor_newWithSize4d(state, size0, -1, -1, -1);
}

THClTensor *THClTensor_newWithSize2d(THClState *state, long size0, long size1)
{
  return THClTensor_newWithSize4d(state, size0, size1, -1, -1);
}

THClTensor *THClTensor_newWithSize3d(THClState *state, long size0, long size1, long size2)
{
  return THClTensor_newWithSize4d(state, size0, size1, size2, -1);
}

THClTensor *THClTensor_newWithSize4d(THClState *state, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THClTensor *self = (THClTensor*)THAlloc(sizeof(THClTensor));
  THClTensor_rawInit(state, self);
  THClTensor_rawResize(state, self, 4, size, NULL);

  return self;
}

THClTensor *THClTensor_newClone(THClState *state, THClTensor *self)
{
  THClTensor *tensor = THClTensor_new(state);
  THClTensor_resizeAs(state, tensor, self);
  THClTensor_copy(state, tensor, self);
  return tensor;
}

THClTensor *THClTensor_newContiguous(THClState *state, THClTensor *self)
{
  if(!THClTensor_isContiguous(state, self))
    return THClTensor_newClone(state, self);
  else
  {
    THClTensor_retain(state, self);
    return self;
  }
}

THClTensor *THClTensor_newSelect(THClState *state, THClTensor *tensor, int dimension_, long sliceIndex_)
{
  THClTensor *self = THClTensor_newWithTensor(state, tensor);
  THClTensor_select(state, self, NULL, dimension_, sliceIndex_);
  return self;
}

THClTensor *THClTensor_newNarrow(THClState *state, THClTensor *tensor, int dimension_, long firstIndex_, long size_)
{
  THClTensor *self = THClTensor_newWithTensor(state, tensor);
  THClTensor_narrow(state, self, NULL, dimension_, firstIndex_, size_);
  return self;
}

THClTensor *THClTensor_newTranspose(THClState *state, THClTensor *tensor, int dimension1_, int dimension2_)
{
  THClTensor *self = THClTensor_newWithTensor(state, tensor);
  THClTensor_transpose(state, self, NULL, dimension1_, dimension2_);
  return self;
}

THClTensor *THClTensor_newUnfold(THClState *state, THClTensor *tensor, int dimension_, long size_, long step_)
{
  THClTensor *self = THClTensor_newWithTensor(state, tensor);
  THClTensor_unfold(state, self, NULL, dimension_, size_, step_);
  return self;
}

/* Resize */
void THClTensor_resize(THClState *state, THClTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

  THClTensor_rawResize(state, self, size->size, size->data, (stride ? stride->data : NULL));
}

void THClTensor_resizeAs(THClState *state, THClTensor *self, THClTensor *src)
{
  int isSame = 0;
  int d;
  if(self->nDimension == src->nDimension)
  {
    isSame = 1;
    for(d = 0; d < self->nDimension; d++)
    {
      if(self->size[d] != src->size[d])
      {
        isSame = 0;
        break;
      }
    }
  }

  if(!isSame)
    THClTensor_rawResize(state, self, src->nDimension, src->size, NULL);
}

void THClTensor_resize1d(THClState *state, THClTensor *tensor, long size0)
{
  THClTensor_resize4d(state, tensor, size0, -1, -1, -1);
}

void THClTensor_resize2d(THClState *state, THClTensor *tensor, long size0, long size1)
{
  THClTensor_resize4d(state, tensor, size0, size1, -1, -1);
}

void THClTensor_resize3d(THClState *state, THClTensor *tensor, long size0, long size1, long size2)
{
  THClTensor_resize4d(state, tensor, size0, size1, size2, -1);
}

void THClTensor_resize4d(THClState *state, THClTensor *self, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THClTensor_rawResize(state, self, 4, size, NULL);
}

void THClTensor_resize5d(THClState *state, THClTensor *self, long size0, long size1, long size2, long size3, long size4)
{
    long size[5] = {size0, size1, size2, size3, size4};

  THClTensor_rawResize(state, self, 5, size, NULL);
}

void THClTensor_set(THClState *state, THClTensor *self, THClTensor *src)
{
  if(self != src)
    THClTensor_rawSet(state,
                        self,
                        src->storage,
                        src->storageOffset,
                        src->nDimension,
                        src->size,
                        src->stride);
}

void THClTensor_setStorage(THClState *state, THClTensor *self, THClStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_)
{
  if(size_ && stride_)
    THArgCheck(size_->size == stride_->size, 5, "inconsistent size/stride sizes");

  THClTensor_rawSet(state,
                      self,
                      storage_,
                      storageOffset_,
                      (size_ ? size_->size : (stride_ ? stride_->size : 0)),
                      (size_ ? size_->data : NULL),
                      (stride_ ? stride_->data : NULL));
}

void THClTensor_setStorage1d(THClState *state, THClTensor *self, THClStorage *storage_, long storageOffset_,
                             long size0_, long stride0_)
{
  THClTensor_setStorage4d(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            -1, -1,
                            -1, -1,
                            -1, -1);
}

void THClTensor_setStorage2d(THClState *state, THClTensor *self, THClStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_)
{
  THClTensor_setStorage4d(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            size1_, stride1_,
                            -1, -1,
                            -1, -1);
}

void THClTensor_setStorage3d(THClState *state, THClTensor *self, THClStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_)
{
  THClTensor_setStorage4d(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            size1_, stride1_,
                            size2_, stride2_,
                            -1, -1);
}

void THClTensor_setStorage4d(THClState *state, THClTensor *self, THClStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_,
                             long size3_, long stride3_)
{

  long size[4] = {size0_, size1_, size2_, size3_};
  long stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THClTensor_rawSet(state, self, storage_, storageOffset_, 4, size, stride);
}


void THClTensor_narrow(THClState *state, THClTensor *self, THClTensor *src, int dimension, long firstIndex, long size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 4, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= src->size[dimension]), 5, "out of range");

  THClTensor_set(state, self, src);

  if(firstIndex > 0)
    self->storageOffset += firstIndex*self->stride[dimension];

  self->size[dimension] = size;
}

void THClTensor_select(THClState *state, THClTensor *self, THClTensor *src, int dimension, long sliceIndex)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 4, "out of range");

  THClTensor_set(state, self, src);
  THClTensor_narrow(state, self, NULL, dimension, sliceIndex, 1);
  for(d = dimension; d < self->nDimension-1; d++)
  {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->nDimension--;
}

void THClTensor_transpose(THClState *state, THClTensor *self, THClTensor *src, int dimension1, int dimension2)
{
  long z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 2, "out of range");

  THClTensor_set(state, self, src);

  if(dimension1 == dimension2) {
    return;
  }

  z = self->stride[dimension1];
  self->stride[dimension1] = self->stride[dimension2];
  self->stride[dimension2] = z;
  z = self->size[dimension1];
  self->size[dimension1] = self->size[dimension2];
  self->size[dimension2] = z;
}

void THClTensor_unfold(THClState *state, THClTensor *self, THClTensor *src, int dimension, long size, long step)
{
  long *newSize;
  long *newStride;
  int d;

  if(!src)
    src = self;

  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->nDimension, 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THClTensor_set(state, self, src);

  newSize = (long*)THAlloc(sizeof(long)*(self->nDimension+1));
  newStride = (long*)THAlloc(sizeof(long)*(self->nDimension+1));

  newSize[self->nDimension] = size;
  newStride[self->nDimension] = self->stride[dimension];
  for(d = 0; d < self->nDimension; d++)
  {
    if(d == dimension)
    {
      newSize[d] = (self->size[d] - size) / step + 1;
      newStride[d] = step*self->stride[d];
    }
    else
    {
      newSize[d] = self->size[d];
      newStride[d] = self->stride[d];
    }
  }

  THFree(self->size);
  THFree(self->stride);

  self->size = newSize;
  self->stride = newStride;
  self->nDimension++;
}

/* we have to handle the case where the result is a number */
void THClTensor_squeeze(THClState *state, THClTensor *self, THClTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THClTensor_set(state, self, src);

  for(d = 0; d < src->nDimension; d++)
  {
    if(src->size[d] != 1)
    {
      if(d != ndim)
      {
        self->size[ndim] = src->size[d];
        self->stride[ndim] = src->stride[d];
      }
      ndim++;
    }
  }

  /* right now, we do not handle 0-dimension tensors */
  if(ndim == 0 && src->nDimension > 0)
  {
    self->size[0] = 1;
    self->stride[0] = 1;
    ndim = 1;
  }
  self->nDimension = ndim;
}

void THClTensor_squeeze1d(THClState *state, THClTensor *self, THClTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(dimension < src->nDimension, 3, "dimension out of range");

  THClTensor_set(state, self, src);

  if(src->size[dimension] == 1 && src->nDimension > 1)
  {
    for(d = dimension; d < self->nDimension-1; d++)
    {
      self->size[d] = self->size[d+1];
      self->stride[d] = self->stride[d+1];
    }
    self->nDimension--;
  }
}

int THClTensor_isContiguous(THClState *state, const THClTensor *self)
{
  long z = 1;
  int d;
  for(d = self->nDimension-1; d >= 0; d--)
  {
    if(self->size[d] != 1)
    {
      if(self->stride[d] == z)
        z *= self->size[d];
      else
        return 0;
    }
  }
  return 1;
}

int THClTensor_isSameSizeAs(THClState *state, const THClTensor *self, const THClTensor* src)
{
  int d;
  if (self->nDimension != src->nDimension)
    return 0;
  for(d = 0; d < self->nDimension; ++d)
  {
    if(self->size[d] != src->size[d])
      return 0;
  }
  return 1;
}

long THClTensor_nElement(THClState *state, const THClTensor *self)
{
  if(self->nDimension == 0)
    return 0;
  else
  {
    long nElement = 1;
    int d;
    for(d = 0; d < self->nDimension; d++)
      nElement *= self->size[d];
    return nElement;
  }
}

void THClTensor_retain(THClState *state, THClTensor *self)
{
  if(self->flag & TH_TENSOR_REFCOUNTED)
    THAtomicIncrementRef(&self->refcount);
}

void THClTensor_free(THClState *state, THClTensor *self)
{
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED)
  {
    if(THAtomicDecrementRef(&self->refcount))
    {
      THFree(self->size);
      THFree(self->stride);
      if(self->storage)
        THClStorage_free(state, self->storage);
      THFree(self);
    }
  }
}

void THClTensor_freeCopyTo(THClState *state, THClTensor *self, THClTensor *dst)
{
  if(self != dst)
    THClTensor_copy(state, dst, self);

  THClTensor_free(state, self);
}

/*******************************************************************************/

static void THClTensor_rawInit(THClState *state, THClTensor *self)
{
  self->refcount = 1;
  self->storage = NULL;
  self->storageOffset = 0;
  self->size = NULL;
  self->stride = NULL;
  self->nDimension = 0;
  self->flag = TH_TENSOR_REFCOUNTED;
}

static void THClTensor_rawSet(THClState *state, THClTensor *self, THClStorage *storage, long storageOffset, int nDimension, long *size, long *stride)
{
  /* storage */
  if(self->storage != storage)
  {
    if(self->storage)
      THClStorage_free(state, self->storage);

    if(storage)
    {
      self->storage = storage;
      THClStorage_retain(state, self->storage);
    }
    else
      self->storage = NULL;
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THClTensor_rawResize(state, self, nDimension, size, stride);
}

static void THClTensor_rawResize(THClState *state, THClTensor *self, int nDimension, long *size, long *stride)
{
  int d;
  int nDimension_;
  long totalSize;
  int hascorrectsize = 1;

  nDimension_ = 0;
  for(d = 0; d < nDimension; d++)
  {
    if(size[d] > 0)
    {
      nDimension_++;
      if((self->nDimension > d) && (size[d] != self->size[d]))
        hascorrectsize = 0;

      if((self->nDimension > d) && stride && (stride[d] >= 0) && (stride[d] != self->stride[d]))
        hascorrectsize = 0;
    }
    else
      break;
  }
  nDimension = nDimension_;

  if(nDimension != self->nDimension)
    hascorrectsize = 0;

  if(hascorrectsize)
    return;

  if(nDimension > 0)
  {
    if(nDimension != self->nDimension)
    {
      self->size = (long*)THRealloc(self->size, sizeof(long)*nDimension);
      self->stride = (long*)THRealloc(self->stride, sizeof(long)*nDimension);
      self->nDimension = nDimension;
    }

    totalSize = 1;
    for(d = self->nDimension-1; d >= 0; d--)
    {
      self->size[d] = size[d];
      if(stride && (stride[d] >= 0) )
        self->stride[d] = stride[d];
      else
      {
        if(d == self->nDimension-1)
          self->stride[d] = 1;
        else
          self->stride[d] = self->size[d+1]*self->stride[d+1];
      }
      totalSize += (self->size[d]-1)*self->stride[d];
    }

    if(totalSize+self->storageOffset > 0)
    {
      if(!self->storage)
        self->storage = THClStorage_new(state);
      if(totalSize+self->storageOffset > self->storage->size)
        THClStorage_resize(state, self->storage, totalSize+self->storageOffset);
    }
  }
  else
    self->nDimension = 0;
}

void THClTensor_set1d(THClState *state, THClTensor *tensor, long x0, float value)
{
  THError("Please convert to FloatTensor, then update, then copy back to GPU");
//  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
//  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
//  THClStorage_set(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0], value);
}

float THClTensor_get1d(THClState *state, const THClTensor *tensor, long x0)
{
  THError("Please convert to FloatTensor, then update, then copy back to GPU");
//  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
//  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
//  return THClStorage_get(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
  return 0;
}

void THClTensor_set2d(THClState *state, THClTensor *tensor, long x0, long x1, float value)
{
  THError("Please convert to FloatTensor, then update, then copy back to GPU");
//  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
//  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
//  THClStorage_set(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1], value);
}

float THClTensor_get2d(THClState *state, const THClTensor *tensor, long x0, long x1)
{
  THError("Please convert to FloatTensor, then update, then copy back to GPU");
//  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
//  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
//  return THClStorage_get(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]);
  return 0;
}

void THClTensor_set3d(THClState *state, THClTensor *tensor, long x0, long x1, long x2, float value)
{
  THError("Please convert to FloatTensor, then update, then copy back to GPU");
//  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
//  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
//  THClStorage_set(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2], value);
}

float THClTensor_get3d(THClState *state, const THClTensor *tensor, long x0, long x1, long x2)
{
  THError("Please convert to FloatTensor, then update, then copy back to GPU");
//  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
//  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
//  return THClStorage_get(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]);
  return 0;
}

void THClTensor_set4d(THClState *state, THClTensor *tensor, long x0, long x1, long x2, long x3, float value)
{
  THError("Please convert to FloatTensor, then update, then copy back to GPU");
//  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
//  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
//  THClStorage_set(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3], value);
}

float THClTensor_get4d(THClState *state, const THClTensor *tensor, long x0, long x1, long x2, long x3)
{
  THError("Please convert to FloatTensor, then update, then copy back to GPU");
//  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
//  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
//  return THClStorage_get(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]);
  return 0;
}
// from .cu
//cudaTextureObject_t THClTensor_getTextureObject(THClState *state, THClTensor *self)
//{
//  THError("THClTensor_getTextureObject Not implemented");
//  return NULL;
//  THAssert(THClTensor_checkGPU(state, 1, self));
//  cudaTextureObject_t texObj;
//  struct cudaResourceDesc resDesc;
//  memset(&resDesc, 0, sizeof(resDesc));
//  resDesc.resType = cudaResourceTypeLinear;
//  resDesc.res.linear.devPtr = THClTensor_data(state, self);
//  resDesc.res.linear.sizeInBytes = THClTensor_nElement(state, self) * 4;
//  resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0,
//                                                  cudaChannelFormatKindFloat);
//  struct cudaTextureDesc texDesc;
//  memset(&texDesc, 0, sizeof(texDesc));
//  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
//  cudaError errcode = cudaGetLastError();
//  if(errcode != cudaSuccess) {
//    if (THClTensor_nElement(state, self) > 2>>27)
//      THError("Failed to create texture object, "
//              "nElement:%ld exceeds 27-bit addressing required for tex1Dfetch. Cl Error: %s",
//              THClTensor_nElement(state, self), cudaGetErrorString(errcode));
//    else
//      THError("Failed to create texture object: %s", cudaGetErrorString(errcode));
//  }
//  return texObj;
//}
// from .cu
THCL_API int THClTensor_getDevice(THClState* state, const THClTensor* thc) {
  return thc->storage->device;
//  THError("THClTensor_getDevice Not implemented");
//  return 0;
//  if (!thc->storage) return -1;
//  cudaPointerAttributes attr;
//  THClCheck(cudaPointerGetAttributes(&attr, thc->storage->data));
//  return attr.device;
}
int THClTensor_checkGPU(THClState *state, unsigned int nTensors, ...)
{
 // THError("THClTensor_checkGPU Not implemented");
//  return 0;
//#ifdef DISABLE_CHECK_GPU
  return 1;  // Disable GPU checks.
//#else
//  int curDev = -1;
//  THClCheck(cudaGetDevice(&curDev));
//  va_list(args);
//  va_start(args, nTensors);
//  int valid = 1;
//  for (unsigned int i = 0; i < nTensors; i++) {
//    THClTensor* tensor = va_arg(args, THClTensor*);
//    if (tensor == NULL) {
//      continue;
//    }
//    int tensorDev = THClTensor_getDevice(state, tensor);
//    if (tensorDev != -1 && tensorDev != curDev) {
//      valid = 0;
//      break;
//    }
//  }
//  va_end(args);
//  return valid;
//#endif
}

std::string THClTensor_toString(THClState *state, const THClTensor *tensor) {
  string res = "";
  res += "THClTensor{";
  res += "size={";
  for( int i = 0; i < tensor->nDimension; i++ ) {
    if(i > 0) {
      res += ",";
    }
    res += easycl::toString(tensor->size[i]);
  }
  res += "},";
  res += "stride={";
  for( int i = 0; i < tensor->nDimension; i++ ) {
    if(i > 0) {
      res += ",";
    }
    res += easycl::toString(tensor->stride[i]);
  }
  res += "},";
  res += "offset=" + easycl::toString(tensor->storageOffset);
  res += ",nElem=" + easycl::toString(THClTensor_nElement(state, tensor));
  res += "}";
  return res;
}

