#include <string>

#include "THClTensorMath.h"
#include "THClGeneral.h"
//#include "THCBlas.h"
#include "THClTensorCopy.h"
//#include "THCTensorRandom.h"
#include "THClApply.h"
//#include "THCReduce.cuh"
#include "THClTensorMathPointwise.h"
#include "THClReduceAll.h"
#include "THClReduce.h"
#include "EasyCL.h"
#include "util/StatefulTimer.h"

using namespace std;

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

class TensorPowOp : public HasOperator1, public HasOperator2, public HasScalars {
public:
  int getNumScalars() const { return 1; }
  float getScalar( int index ) const { return val; }
  TensorPowOp(float v) : val(v) {}
  string operator2() const {
    int valint = (int)val;
    if( (float)valint == val ) {
      return "*out = pown(*in1, val1)";
    } else {
      return "*out = pow(*in1, val1)";
    }
  }
  string operator1() const {
    int valint = (int)val;
    if( (float)valint == val ) {
      return "*out = pown(*out, val1)";
    } else {
      return "*out = pow(*out, val1)";
    }
  }
  const float val;
};

void THClTensor_pow(THClState *state, THClTensor *self_, THClTensor *src, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    TensorPowOp op(value);
    if (!THClTensor_pointwiseApply1(state, self_, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src);

    TensorPowOp op(value);
    if (!THClTensor_pointwiseApply2(state, self_, src, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

class TensorTPowOp : public HasOperator2, public HasOperator1, public HasScalars {
public:
  int getNumScalars() const { return 1; }
  float getScalar( int index ) const { return val; }
  TensorTPowOp(float v) : val(v) {}
  string operator2() const {
    return "*out = pow(val1, *in1)";
  }
  string operator1() const {
    return "*out = pow(val1, *out)";
  }
  const float val;
};

void THClTensor_tpow(THClState *state, THClTensor *self_, float value, THClTensor *src)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    TensorTPowOp op(value);
    if (!THClTensor_pointwiseApply1(state, self_, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src);

    TensorTPowOp op(value);
    if (!THClTensor_pointwiseApply2(state, self_, src, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

class TensorATan2Op : public HasOperator2, public HasOperator3 {
public:
  string operator2() const {
    return "*out = atan2(*out, *in1)";
  }
  string operator3() const {
    return "*out = atan2(*in1, *in2)";
  }
};

void THClTensor_atan2(THClState *state, THClTensor *self_, THClTensor *tx, THClTensor *ty)
{
  THAssert(THClTensor_checkGPU(state, 3, self_, tx, ty));
  THArgCheck(THClTensor_nElement(state, tx) ==
             THClTensor_nElement(state, ty), 3, "sizes do not match");
  THClTensor_resizeAs(state, self_, tx);
  TensorATan2Op op;
  if (!THClTensor_pointwiseApply3(state, self_, tx, ty, &op)) {
    THArgCheck(false, 2, CLTORCH_DIM_WARNING);
  }
}

class TensorClampOp : public HasOperator1, public HasOperator2, public HasScalars {
public:
  int getNumScalars() const { return 2; }
  float getScalar( int index ) const {
    if( index == 0 ) { return minValue; }
    return maxValue;
  }
  TensorClampOp(float min, float max) : minValue(min), maxValue(max) {}
  string operator2() const {
    return "*out = fmax(fmin(*in1, val2), val1)";
  }
  string operator1() const {
    return "*out = fmax(fmin(*out, val2), val1)";
  }

  const float minValue;
  const float maxValue;
};

void THClTensor_clamp(THClState *state, THClTensor *self_, THClTensor *src, float min_value,
  float max_value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    TensorClampOp op(min_value, max_value);
    if (!THClTensor_pointwiseApply1(state, self_, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src);

    TensorClampOp op(min_value, max_value);
    if (!THClTensor_pointwiseApply2(state, self_, src, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

class TensorSignOp : public HasOperator1, public HasOperator2 {
  string operator1() const {
    return "*out = (*out > 0) - (*out < 0)";
  }
  string operator2() const {
    return "*out = (*in1 > 0) - (*in1 < 0)";
  }
};

void THClTensor_sign(THClState *state, THClTensor *self_, THClTensor *src)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src));
  TensorSignOp op;
  if (self_ == src) {
    if (!THClTensor_pointwiseApply1(state, self_, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src);
    if (!THClTensor_pointwiseApply2(state, self_, src, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

float THClTensor_meanall(THClState *state, THClTensor *self)
{
  THAssert(THClTensor_checkGPU(state, 1, self));
  THArgCheck(self->nDimension > 0, 1, "empty Tensor");
  return THClTensor_sumall(state, self) / THClTensor_nElement(state, self);
}

void
THClTensor_mean(THClState *state, THClTensor *self, THClTensor *src, long dim)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  THClTensor_sum(state, self, src, dim);
  THClTensor_div(state, self, self, THClTensor_size(state, src, dim));
}

/*
struct square_functor
{
  const float mean;

  square_functor(float mean_) : mean(mean_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return (x-mean)*(x-mean);
  }
};

float THClTensor_varall(THClState *state, THClTensor *self)
{
  THAssert(THClTensor_checkGPU(state, 1, self));
  self = THClTensor_newContiguous(state, self);
  long size = THClTensor_nElement(state, self);
  thrust::device_ptr<float> self_data(THClTensor_data(state, self));

  float mean = THClTensor_meanall(state, self);
  float result = thrust::transform_reduce(self_data, self_data+size, square_functor(mean), (float)0, thrust::plus<float>());

  result = result/(THClTensor_nElement(state, self)-1);

  THClTensor_free(state, self);
  return result;
}

float THClTensor_stdall(THClState *state, THClTensor *self)
{
  THAssert(THClTensor_checkGPU(state, 1, self));
  return sqrt(THClTensor_varall(state, self));
}

template<bool apply_sqrt>
__host__ void THClTensor_varOuterDim(THClState *state, THClTensor *tgt, THClTensor *src, long dimension, int flag)
{
  unsigned ndim = THClTensor_nDimension(state, src);
  // Treat all outer dimensions (i.e. dim < dimension) as one.
  unsigned num_orows = 1;
  for (unsigned dim = 0; dim < dimension; dim++) {
    num_orows *= THClTensor_size(state, src, dim);
  }
  unsigned row_size = THClTensor_size(state, src, dimension);
  // Treat all inner dimensions (i.e. dim > dimension) as one.
  unsigned num_irows = 1;
  for (unsigned dim = dimension + 1; dim < ndim; dim++) {
    num_irows *= THClTensor_size(state, src, dim);
  }

  dim3 threads(min(512, num_irows));
  unsigned maxGridDim = 1024;
  dim3 grid(min(maxGridDim, num_orows), min(maxGridDim, DIVUP(num_irows, threads.x)));

  if (flag) {
    THClTensor_kernel_varOuterDim<true, apply_sqrt><<<grid, threads, 0, THClState_getCurrentStream(state)>>>(
        THClTensor_data(state, tgt), THClTensor_data(state, src), num_orows, num_irows, row_size);
  } else {
    THClTensor_kernel_varOuterDim<false, apply_sqrt><<<grid, threads, 0, THClState_getCurrentStream(state)>>>(
        THClTensor_data(state, tgt), THClTensor_data(state, src), num_orows, num_irows, row_size);
  }
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}
*/
/*
template<bool apply_sqrt>
__host__ void THClTensor_varInnermostDim(THClState *state, THClTensor *tgt, THClTensor *src, int flag)
{
  unsigned ndim = THClTensor_nDimension(state, src);
  // Treat all outer dimensions as a single dimension.
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THClTensor_size(state, src, dim);
  }
  unsigned row_size = THClTensor_size(state, src, ndim - 1);

  // From limited testing, 16x32 seemed a good compromise for handling both long and short dimensions.
  dim3 threads(16, 32);
  dim3 grid(min(1024, DIVUP(num_rows, threads.y)));

  if (flag) {
    THClTensor_kernel_varInnermostDim<true, apply_sqrt><<<grid, threads, 0, THClState_getCurrentStream(state)>>>(
        THClTensor_data(state, tgt), THClTensor_data(state, src), num_rows, row_size);
  } else {
    THClTensor_kernel_varInnermostDim<false, apply_sqrt><<<grid, threads, 0, THClState_getCurrentStream(state)>>>(
        THClTensor_data(state, tgt), THClTensor_data(state, src), num_rows, row_size);
  }
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

void THClTensor_var(THClState *state, THClTensor *self_, THClTensor *src, long dimension, int flag)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src));
  THLongStorage *dim = THClTensor_newSizeOf(state, src);
  THLongStorage_set(dim, dimension, 1);
  THClTensor_resize(state, self_, dim, NULL);
  THLongStorage_free(dim);

  THClTensor *self = THClTensor_newContiguous(state, self_);
  src = THClTensor_newContiguous(state, src);

  if (dimension == THClTensor_nDimension(state, src) - 1) {
    THClTensor_varInnermostDim<false>(state, self, src, flag);
  } else {
    THClTensor_varOuterDim<false>(state, self, src, dimension, flag);
  }

  THClTensor_free(state, src);
  THClTensor_freeCopyTo(state, self, self_);
}

void THClTensor_std(THClState *state, THClTensor *self_, THClTensor *src, long dimension, int flag)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src));
  THLongStorage *dim = THClTensor_newSizeOf(state, src);
  THLongStorage_set(dim, dimension, 1);
  THClTensor_resize(state, self_, dim, NULL);
  THLongStorage_free(dim);

  THClTensor *self = THClTensor_newContiguous(state, self_);
  src = THClTensor_newContiguous(state, src);

  if (dimension == THClTensor_nDimension(state, src) - 1) {
    THClTensor_varInnermostDim<true>(state, self, src, flag);
  } else {
    THClTensor_varOuterDim<true>(state, self, src, dimension, flag);
  }

  THClTensor_free(state, src);
  THClTensor_freeCopyTo(state, self, self_);
}
*/

class NormFunctor : public HasOperator2
{
public:
  const float exponent;

  NormFunctor(float exponent_) : exponent(exponent_) {}
  std::string operator2() const
  {
    return "*out = pow(fabs(*in1), " + easycl::toString(exponent) + ")";
  }
};

struct PartialNotEqualFunctor : public HasOperator2
{
  const float rhs;
  PartialNotEqualFunctor(float rhs) : rhs(rhs) {}
  std::string operator2() const
  {
    return "*out = *in1 != " + easycl::toString(rhs);
  }
};

float THClTensor_normall(THClState *state, THClTensor *self, float value)
{
  THAssert(THClTensor_checkGPU(state, 1, self));
  
  float result;
  if(value == 0.0f) {
    PartialNotEqualFunctor modifyOp(0.0f);
    TensorAddOp reduceOp;
    const int device = self->storage->device;
    THClScratchSpace *scratch = THClState_getDeviceScratchSpace(state, device, 0);
    if (!THClTensor_reduceAll(state, self,
          &modifyOp,
          &reduceOp,
          0.0f, scratch->wrapper)) {
      THArgCheck(false, 1, CLTORCH_DIM_WARNING);
    }
    StatefulTimer::timeCheck("normall before copytohost");
    scratch->wrapper->copyToHost();
    StatefulTimer::timeCheck("normall after copytohost");
    result = scratch->data[0];
  } else {
    NormFunctor modifyOp(value);
    TensorAddOp reduceOp;
    const int device = self->storage->device;
    THClScratchSpace *scratch = THClState_getDeviceScratchSpace(state, device, 0);
    if (!THClTensor_reduceAll(state, self,
          &modifyOp,
          &reduceOp,
          0.0f, scratch->wrapper)) {
      THArgCheck(false, 1, CLTORCH_DIM_WARNING);
    }
    StatefulTimer::timeCheck("normall before copytohost");
    scratch->wrapper->copyToHost();
    StatefulTimer::timeCheck("normall after copytohost");
    result = scratch->data[0];
    result = pow(result, (float)1.0/value);
  }

  return result;
}

void THClTensor_norm(THClState *state, THClTensor* self, THClTensor* src, float value, long dimension)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  TensorAddOp addOp;
  if (value == 0.0f) {
    PartialNotEqualFunctor partialNotEqualFunctor(0.0f);
    THClTensor_reduceDim(state, self, src,
                         0.0f,
                         &partialNotEqualFunctor, &addOp,
                         dimension);
  } else {
    NormFunctor normFunctor(value);
    THClTensor_reduceDim(state, self, src,
                         0.0f,
                         &normFunctor, &addOp,
                         dimension);
    THClTensor_pow(state, self, self, 1/value);
  }
}

/*
void THClTensor_renorm(THClState *state, THClTensor* self, THClTensor* src, float value, long dimension, float maxnorm)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  THClTensor *self_;
  THClTensor *src_ = THClTensor_newTranspose(state, src, dimension, 0);
  THClTensor *data = THClTensor_newClone(state, src_);
  long size = THClTensor_nElement(state, data)/data->size[0];

  THArgCheck(dimension >= 0 && dimension < THClTensor_nDimension(state, src), 3, "invalid dimension");
  THArgCheck(value > 0, 2, "non-positive-norm not supported");
  THArgCheck(THClTensor_nDimension(state, src) > 1, 1, "need at least 2 dimensions");

  dim3 grid(data->size[0]);
  dim3 threads(32);

  THClTensor_kernel_renorm<<<grid, threads, 0, THClState_getCurrentStream(state)>>>(THClTensor_data(state, data), value, size, maxnorm);

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THClTensor_free(state, src_);
  self_ = THClTensor_newTranspose(state, data, dimension, 0);
  THClTensor_resizeAs(state, self, self_);
  THClTensor_freeCopyTo(state, self_, self);
  THClTensor_free(state, data);
}

struct dist_functor
{
  const float exponent;

  dist_functor(float exponent_) : exponent(exponent_) {}

  __host__ __device__ float operator()(const float& x, const float& y) const
  {
    return pow(fabs(x-y), exponent);
  }
};

float THClTensor_dist(THClState *state, THClTensor *self, THClTensor *src, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  self = THClTensor_newContiguous(state, self);
  long size = THClTensor_nElement(state, self);
  src = THClTensor_newContiguous(state, src);
  thrust::device_ptr<float> self_data(THClTensor_data(state, self));
  thrust::device_ptr<float> src_data(THClTensor_data(state, src));

  float result = thrust::inner_product(self_data, self_data+size, src_data, (float) 0,thrust::plus<float>(), dist_functor(value));

  THClTensor_free(state, src);
  THClTensor_free(state, self);

  return pow(result, (float)1.0/value);
}

void THClTensor_rand(THClState *state, THClTensor *r_, THLongStorage *size)
{
  THAssert(THClTensor_checkGPU(state, 1, r_));
  THClTensor_resize(state, r_, size, NULL);
  THClTensor_uniform(state, r_, 0, 1);
}

void THClTensor_randn(THClState *state, THClTensor *r_, THLongStorage *size)
{
  THAssert(THClTensor_checkGPU(state, 1, r_));
  THClTensor_resize(state, r_, size, NULL);
  THClTensor_normal(state, r_, 0, 1);
}
*/
