// from lib/THC/THCTensorMathTransformReduce.cu:

#include "THClTensorMath.h"
#include "THClGeneral.h"
#include "THClBlas.h"
#include "THClTensorCopy.h"
#include "THClTensorRandom.h"
#include "THClApply.cuh"
#include "THClReduce.cuh"

// #include <thrust/device_ptr.h>
// #include <thrust/fill.h>
// #include <thrust/functional.h>
// #include <thrust/reduce.h>
// #include <thrust/inner_product.h>

//template<class BinaryFunction>
//__host__ void THClTensor_transformReduceOuterDimIndex(THClState *state, THClTensor *tgt1, THClTensor *tgt2,
//                                                   THClTensor *src,
////                                                    long rdim, thrust::pair<float,float> init,
//                                                   BinaryFunction binary_op)
//{
//  unsigned ndim = THClTensor_nDimension(state, src);
//  unsigned num_orows = 1;
//  for (unsigned dim = 0; dim < rdim; dim++) {
//    num_orows *= THClTensor_size(state, src, dim);
//  THError("Not implemented");
//  }
//  unsigned row_size = THClTensor_size(state, src, rdim);
//  unsigned num_irows = 1;
//  for (unsigned dim = rdim + 1; dim < ndim; dim++) {
//    num_irows *= THClTensor_size(state, src, dim);
//  }

//  dim3 threads(min(512, num_irows));
//  unsigned maxGridDim = 1024;
//  dim3 grid(min(maxGridDim, num_orows), min(maxGridDim, THCCeilDiv(num_irows, threads.x)));

//  THClTensor_kernel_transformReduceOuterDimIndex<<<grid, threads, 0, THClState_getCurrentStream(state)>>>(
//    THClTensor_data(state, tgt1), THClTensor_data(state, tgt2),
//    THClTensor_data(state, src), num_orows, num_irows, row_size, init, binary_op);
//  cudaError errcode = cudaGetLastError();
//  if(errcode != cudaSuccess) {
//    THError(cudaGetErrorString(errcode));
//  }
//}

//template<class BinaryFunction>
//__host__ void THClTensor_transformReduceInnermostDimIndex(
//  THClState *state, THClTensor *tgt1, THClTensor *tgt2, THClTensor *src,
////   thrust::pair<float,float> init, BinaryFunction binary_op)
//{
//  unsigned ndim = THClTensor_nDimension(state, src);
//  unsigned num_rows = 1;
//  for (unsigned dim = 0; dim < ndim - 1; dim++) {
//    num_rows *= THClTensor_size(state, src, dim);
//  THError("Not implemented");
//  }
//  unsigned row_size = THClTensor_size(state, src, ndim - 1);

//  dim3 threads(16, 32);
//  dim3 grid(min(1024, THCCeilDiv(num_rows, threads.y)));

//  THClTensor_kernel_transformReduceInnermostDimIndex<<<grid, threads, 0, THClState_getCurrentStream(state)>>>(
//    THClTensor_data(state, tgt1), THClTensor_data(state, tgt2),
//    THClTensor_data(state, src), num_rows, row_size, init, binary_op);
//  cudaError errcode = cudaGetLastError();
//  if(errcode != cudaSuccess) {
//    THError(cudaGetErrorString(errcode));
//  }
//}

//template<class BinaryFunction>
//void THClTensor_reduceDimIndex(THClState *state, THClTensor *tgt1_, THClTensor *tgt2_, THClTensor *src,
////                              long dimension, thrust::pair<float,float> init,
//                                     BinaryFunction binary_op)
//{
//  THArgCheck(dimension >= 0 && dimension < THClTensor_nDimension(state, src), 3, "dimension out of range");

//  THLongStorage *dim = THClTensor_newSizeOf(state, src);
//  THLongStorage_set(dim, dimension, 1);
//  THClTensor_resize(state, tgt1_, dim, NULL);
//  THClTensor_resize(state, tgt2_, dim, NULL);
//  THLongStorage_free(dim);

//  THClTensor *tgt1 = THClTensor_newContiguous(state, tgt1_);
//  THClTensor *tgt2 = THClTensor_newContiguous(state, tgt2_);
//  src = THClTensor_newContiguous(state, src);

//  if(dimension == THClTensor_nDimension(state, src)-1) {
//    THClTensor_transformReduceInnermostDimIndex(state, tgt1, tgt2, src, init, binary_op);
//  THError("Not implemented");
//  } else {
//    THClTensor_transformReduceOuterDimIndex(state, tgt1, tgt2, src, dimension, init, binary_op);
//  }

//  THClTensor_free(state, src);
//  THClTensor_freeCopyTo(state, tgt1, tgt1_);
//  THClTensor_freeCopyTo(state, tgt2, tgt2_);
//}

//void THClTensor_max(THClState *state, THClTensor *values, THClTensor *indices, THClTensor *src, long dimension)
//{
//  THAssert(THClTensor_checkGPU(state, 3, values, indices, src));
//  const float minfloat32 = -3.402823466e+38f;
////   thrust::pair<float,float> init = thrust::make_pair<float,float>(minfloat32, -1);
//  THError("Not implemented");
//  return 0;
//  //   return THClTensor_reduceDimIndex(state, values, indices, src, dimension, init,
//                                 maxvalue_functor());
//}

//void THClTensor_min(THClState *state, THClTensor *values, THClTensor *indices, THClTensor *src, long dimension)
//{
//  THAssert(THClTensor_checkGPU(state, 3, values, indices, src));
//  const float maxfloat32 = 3.402823466e+38f;
////   thrust::pair<float,float> init = thrust::make_pair<float,float>(maxfloat32, -1);
//  THError("Not implemented");
//  return 0;
//  //   return THClTensor_reduceDimIndex(state, values, indices, src, dimension, init,
//                                     minvalue_functor());
//}

