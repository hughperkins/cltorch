#ifndef THCL_TENSOR_INC
#define THCL_TENSOR_INC

#include <stdint.h>

#include "THTensor.h"
#include "THClStorage.h"
#include "THClGeneral.h"

#define TH_TENSOR_REFCOUNTED 1

//struct CLWrapper;

typedef struct THClTensor
{
    long *size;
    long *stride;
    int nDimension;

    THClStorage *storage;
    long storageOffset;
    int refcount;

    char flag;

    int device;
} THClTensor;


/**** access methods ****/
THCL_API THClStorage* THClTensor_storage(THClState *state, const THClTensor *self);
THCL_API long THClTensor_storageOffset(THClState *state, const THClTensor *self);
THCL_API int THClTensor_nDimension(THClState *state, const THClTensor *self);
THCL_API long THClTensor_size(THClState *state, const THClTensor *self, int dim);
THCL_API long THClTensor_stride(THClState *state, const THClTensor *self, int dim);
THCL_API THLongStorage *THClTensor_newSizeOf(THClState *state, THClTensor *self);
THCL_API THLongStorage *THClTensor_newStrideOf(THClState *state, THClTensor *self);
THCL_API float *THClTensor_data(THClState *state, const THClTensor *self);
#ifdef __cplusplus
THCL_API class CLWrapper *THClTensor_wrapper(THClState *state, const THClTensor *self);
#endif // __cplusplus

THCL_API void THClTensor_setFlag(THClState *state, THClTensor *self, const char flag);
THCL_API void THClTensor_clearFlag(THClState *state, THClTensor *self, const char flag);


/**** creation methods ****/
THCL_API THClTensor *THClTensor_new(THClState *state) DEPRECATED_POST;
THCL_API THClTensor *THClTensor_newv2(THClState *state, int device);
THCL_API THClTensor *THClTensor_newWithTensor(THClState *state, THClTensor *tensor);
/* stride might be NULL */
THCL_API THClTensor *THClTensor_newWithStorage(THClState *state, int device, THClStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THCL_API THClTensor *THClTensor_newWithStorage1d(THClState *state, int device, THClStorage *storage_, long storageOffset_,
                                long size0_, long stride0_);
THCL_API THClTensor *THClTensor_newWithStorage2d(THClState *state, int device, THClStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
THCL_API THClTensor *THClTensor_newWithStorage3d(THClState *state, int device, THClStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
THCL_API THClTensor *THClTensor_newWithStorage4d(THClState *state, int device, THClStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);

/* stride might be NULL */
THCL_API THClTensor *THClTensor_newWithSize(THClState *state, int device, THLongStorage *size_, THLongStorage *stride_);
THCL_API THClTensor *THClTensor_newWithSize1d(THClState *state, int device, long size0_);
THCL_API THClTensor *THClTensor_newWithSize2d(THClState *state, int device, long size0_, long size1_);
THCL_API THClTensor *THClTensor_newWithSize3d(THClState *state, int device, long size0_, long size1_, long size2_);
THCL_API THClTensor *THClTensor_newWithSize4d(THClState *state, int device, long size0_, long size1_, long size2_, long size3_);

THCL_API THClTensor *THClTensor_newClone(THClState *state, THClTensor *self);
THCL_API THClTensor *THClTensor_newContiguous(THClState *state, THClTensor *tensor);
THCL_API THClTensor *THClTensor_newSelect(THClState *state, THClTensor *tensor, int dimension_, long sliceIndex_);
THCL_API THClTensor *THClTensor_newNarrow(THClState *state, THClTensor *tensor, int dimension_, long firstIndex_, long size_);
THCL_API THClTensor *THClTensor_newTranspose(THClState *state, THClTensor *tensor, int dimension1_, int dimension2_);
THCL_API THClTensor *THClTensor_newUnfold(THClState *state, THClTensor *tensor, int dimension_, long size_, long step_);

THCL_API void THClTensor_resize(THClState *state, THClTensor *tensor, THLongStorage *size, THLongStorage *stride);
THCL_API void THClTensor_resizeAs(THClState *state, THClTensor *tensor, THClTensor *src);
THCL_API void THClTensor_resize0d(THClState *state, THClTensor *tensor);
THCL_API void THClTensor_resize1d(THClState *state, THClTensor *tensor, long size0_);
THCL_API void THClTensor_resize2d(THClState *state, THClTensor *tensor, long size0_, long size1_);
THCL_API void THClTensor_resize3d(THClState *state, THClTensor *tensor, long size0_, long size1_, long size2_);
THCL_API void THClTensor_resize4d(THClState *state, THClTensor *tensor, long size0_, long size1_, long size2_, long size3_);
THCL_API void THClTensor_resize5d(THClState *state, THClTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

THCL_API void THClTensor_set(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_setStorage(THClState *state, THClTensor *self, THClStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THCL_API void THClTensor_setStorage1d(THClState *state, THClTensor *self, THClStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_);
THCL_API void THClTensor_setStorage2d(THClState *state, THClTensor *self, THClStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_);
THCL_API void THClTensor_setStorage3d(THClState *state, THClTensor *self, THClStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_);
THCL_API void THClTensor_setStorage4d(THClState *state, THClTensor *self, THClStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_,
                                    long size3_, long stride3_);

THCL_API void THClTensor_narrow(THClState *state, THClTensor *self, THClTensor *src, int dimension_, long firstIndex_, long size_);
THCL_API void THClTensor_select(THClState *state, THClTensor *self, THClTensor *src, int dimension_, long sliceIndex_);
THCL_API void THClTensor_transpose(THClState *state, THClTensor *self, THClTensor *src, int dimension1_, int dimension2_);
THCL_API void THClTensor_unfold(THClState *state, THClTensor *self, THClTensor *src, int dimension_, long size_, long step_);

THCL_API void THClTensor_squeeze(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_squeeze1d(THClState *state, THClTensor *self, THClTensor *src, int dimension_);

THCL_API int THClTensor_isContiguous(THClState *state, const THClTensor *self);
THCL_API int THClTensor_isSameSizeAs(THClState *state, const THClTensor *self, const THClTensor *src);
THCL_API long THClTensor_nElement(THClState *state, const THClTensor *self);

THCL_API void THClTensor_retain(THClState *state, THClTensor *self);
THCL_API void THClTensor_free(THClState *state, THClTensor *self);
THCL_API void THClTensor_freeCopyTo(THClState *state, THClTensor *self, THClTensor *dst);

/* Slow access methods [check everything] */
THCL_API void THClTensor_set1d(THClState *state, THClTensor *tensor, long x0, float value);
THCL_API void THClTensor_set2d(THClState *state, THClTensor *tensor, long x0, long x1, float value);
THCL_API void THClTensor_set3d(THClState *state, THClTensor *tensor, long x0, long x1, long x2, float value);
THCL_API void THClTensor_set4d(THClState *state, THClTensor *tensor, long x0, long x1, long x2, long x3, float value);

THCL_API float THClTensor_get1d(THClState *state, const THClTensor *tensor, long x0);
THCL_API float THClTensor_get2d(THClState *state, const THClTensor *tensor, long x0, long x1);
THCL_API float THClTensor_get3d(THClState *state, const THClTensor *tensor, long x0, long x1, long x2);
THCL_API float THClTensor_get4d(THClState *state, const THClTensor *tensor, long x0, long x1, long x2, long x3);

/* GPU-specific functions */
//THCL_API cudaTextureObject_t THClTensor_getTextureObject(THClState *state, THClTensor *self);
THCL_API int THClTensor_getDevice(THClState *state, const THClTensor *self);
THCL_API int THClTensor_checkGPU(THClState *state, unsigned int nTensors, ...);

// new
#ifdef __cplusplus
THCL_API_CPP std::string THClTensor_toString(THClState *state, const THClTensor *tensor);
THCL_API EasyCL *THClTensor_getCl(THClState *state, const THClTensor *tensor);
#endif // __cplusplus
THCL_API int THClTensor_getDevice(THClState *state, const THClTensor *tensor);

#endif
