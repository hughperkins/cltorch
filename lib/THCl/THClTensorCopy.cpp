#include "THClApply.h"

//static inline int curGPU() {
//  int curDev;
//  THClCheck(cudaGetDevice(&curDev));
//  return curDev;
//}

THCL_API void
THClTensor_copy(THClState* state, THClTensor* dst, THClTensor* src) {
  THError("Not implemented");
//  long totalElements = THClTensor_nElement(state, dst);

//  THArgCheck(totalElements == THClTensor_nElement(state, src), 2,
//             "sizes do not match");

//  if (THClTensor_nDimension(state, dst) == 0) {
//    // Zero-dim tensor; copy nothing
//    return;
//  }

//  // We can memcpy the memory if:
//  // -both tensors are contiguous; or,
//  // -there is only one element to copy; or,
//  // -FIXME: if both tensors have matching size and stride arrays, and no
//  // holes within (in other words, there is some permutation that can be applied
//  // to the size/strides such that the resulting tensor is contiguous).
//  bool srcContig = THClTensor_isContiguous(state, src);
//  bool dstContig = THClTensor_isContiguous(state, dst);
//  bool memcpyEligible = (srcContig && dstContig) || (totalElements == 1);

//  if (memcpyEligible) {
//    THClCheck(cudaMemcpyAsync(THClTensor_data(state, dst),
//                                THClTensor_data(state, src),
//                                totalElements * sizeof(float),
//                                cudaMemcpyDeviceToDevice,
//                                THClState_getCurrentStream(state)));
//  } else {
//    int oldDev = curGPU();
//    int srcDev = THClTensor_getDevice(state, src);
//    int dstDev = THClTensor_getDevice(state, dst);

//    if (srcDev == dstDev) {
//      if (oldDev != srcDev) {
//        THClCheck(cudaSetDevice(srcDev));
//      }

//      bool succ =
//        THClTensor_pointwiseApply2(state, dst, src, CopyOp<float>());
//      THArgCheck(succ, 2, CUTORCH_DIM_WARNING);
//    } else { // multi-gpu
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
//      THArgCheck(succ, 2, CUTORCH_DIM_WARNING);

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
//    }

//    if (curGPU() != oldDev) {
//      THClCheck(cudaSetDevice(oldDev));
//    }
//  }

//  cudaError errcode = cudaGetLastError();
//  if (errcode != cudaSuccess) {
//    THError(cudaGetErrorString(errcode));
//  }
}
