#include <stdexcept>
#include <iostream>

#include "THClApply.h"
#include "THClTensorCopy.h"
#include "THClGeneral.h"
#include "THClTensor.h"

#include "EasyCL.h"

using namespace std;

// ============ from .c:

/* specific methods */

void THClTensor_copyFloat(THClState *state, THClTensor *self, struct THFloatTensor *src)
{
  THArgCheck(THClTensor_nElement(state, self) == THFloatTensor_nElement(src), 2, "sizes do not match"); 
  {
    THClTensor *selfc = THClTensor_newContiguous(state, self);
    src = THFloatTensor_newContiguous(src);
  
    int numElements = THFloatTensor_nElement(src);
    float *dest_segment = selfc->storage->data + selfc->storageOffset;
    float *src_segment = src->storage->data + src->storageOffset;
    for( int i = 0; i < numElements; i++ ) {
      dest_segment[i] = src_segment[i];
    }
    selfc->storage->wrapper->copyToDevice();

    THFloatTensor_free(src);
    THClTensor_freeCopyTo(state, selfc, self);
  }
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
  THArgCheck(THFloatTensor_nElement(self) == THClTensor_nElement(state, src), 2, "sizes do not match");

  {
    THFloatTensor *selfc = THFloatTensor_newContiguous(self);
    src = THClTensor_newContiguous(state, src);

    int numElements = THClTensor_nElement(state, src);
    if( src->storage->wrapper->isDeviceDirty() ) {
        src->storage->wrapper->copyToHost();
    }
    float *dest_segment = selfc->storage->data + selfc->storageOffset;
    float *src_segment =  src->storage->data + src->storageOffset;
    for( int i = 0; i < numElements; i++ ) {
        dest_segment[i] = src_segment[i];
    }

    THClTensor_free(state, src);
    THFloatTensor_freeCopyTo(selfc, self);
  }
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

// ============ from cu:

static inline int curGPU(THClState *state) {
//  int curDev;
//  THClCheck(cudaGetDevice(&curDev));
  return state->currentDevice;
}

THCL_API void
THClTensor_copy(THClState* state, THClTensor* dst, THClTensor* src) {
  long totalElements = THClTensor_nElement(state, dst);

  THArgCheck(totalElements == THClTensor_nElement(state, src), 2,
             "sizes do not match");

  if (THClTensor_nDimension(state, dst) == 0) {
    // Zero-dim tensor; copy nothing
    return;
  }

  // We can memcpy the memory if:
  // -both tensors are contiguous; or,
  // -there is only one element to copy; or,
  // -FIXME: if both tensors have matching size and stride arrays, and no
  // holes within (in other words, there is some permutation that can be applied
  // to the size/strides such that the resulting tensor is contiguous).
  bool srcContig = THClTensor_isContiguous(state, src);
  bool dstContig = THClTensor_isContiguous(state, dst);
  bool memcpyEligible = (srcContig && dstContig) || (totalElements == 1);

  if (memcpyEligible) {
    if( !dst->storage->wrapper->isOnDevice() ) {
      dst->storage->wrapper->createOnDevice();
    }
    src->storage->wrapper->copyTo( dst->storage->wrapper );
 } else {
    int oldDev = curGPU(state);
    int srcDev = THClTensor_getDevice(state, src);
    int dstDev = THClTensor_getDevice(state, dst);

    if (srcDev == dstDev) {
      if (oldDev != srcDev) {
        cout << "srcDev=" << srcDev << " dstDev=" << dstDev << endl;
        THError("Not implemented");
//        THClCheck(cudaSetDevice(srcDev));
      }

      CopyOp copyOp;
      bool succ =
        THClTensor_pointwiseApply2(state, dst, src, *&copyOp );
      THArgCheck(succ, 2, CLTORCH_DIM_WARNING);
    } else { // multi-gpu
//      // empirically, running the kernel on the device that holds the
//      // non-contiguous tensor is faster by 5-10x
//      int copyDev   = dstContig ? srcDev : dstDev;
//      int remoteDev = dstContig ? dstDev : srcDev;

//      // synchronize remote device before copy
//      cudaEvent_t dataReady;
//      THClCheck(cudaSetDevice(remoteDev));
//      THClCheck(cudaEventCreate(&dataReady));
//      THClCheck(cudaEventRecord(
//                    dataReady,
//                    THClState_getDeviceStream(state, remoteDev, THClState_getCurrentStreamIndex(state))));
//      THClCheck(cudaSetDevice(copyDev));
//      THClCheck(cudaStreamWaitEvent(
//                    THClState_getDeviceStream(state, copyDev, THClState_getCurrentStreamIndex(state)),
//                    dataReady, 0));
//      THClCheck(cudaEventDestroy(dataReady));

//      bool succ =
//        THClTensor_pointwiseApply2(state, dst, src, CopyOp<float>());
//      THArgCheck(succ, 2, CLTORCH_DIM_WARNING);

//      // synchronize remote device after copy
//      cudaEvent_t doneCopying;
//      THClCheck(cudaEventCreate(&doneCopying));
//      THClCheck(cudaEventRecord(
//                    doneCopying,
//                    THClState_getDeviceStream(state, copyDev, THClState_getCurrentStreamIndex(state))));
//      THClCheck(cudaSetDevice(remoteDev));
//      THClCheck(cudaStreamWaitEvent(
//                    THClState_getDeviceStream(state, remoteDev, THClState_getCurrentStreamIndex(state)),
//                    doneCopying, 0));
//      THClCheck(cudaEventDestroy(doneCopying));
      THError("Not implemented");
    }

    if (curGPU(state) != oldDev) {
      state->currentDevice = oldDev;
//      THClCheck(cudaSetDevice(oldDev));
//      THError("Not implemented");
    }
//    throw runtime_error("not implemented");
  }
}
