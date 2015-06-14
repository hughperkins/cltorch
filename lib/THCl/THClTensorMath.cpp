#include <string>
#include "THClTensorMath.h"
#include "THClGeneral.h"
//#include "THClBlas.h"
#include "THClTensorCopy.h"
//#include "THClTensorRandom.h"
#include "THClApply.h"
#include "THClReduce.h"
#include "THClReduceApplyUtils.h"
#include "THClTensorMathPointwise.h"
#include "THClReduceAll.h"

using namespace std;

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

class TensorFillOp : public HasOperator1, public HasScalars {
public:
  int getNumScalars() const { return 1; }
  float getScalar(int index) const { return val; }
  TensorFillOp(float v) : val(v) {}
  string operator1() const {
    return "*out = val1";
  }

  const float val;
};

void THClTensor_fill(THClState* state, THClTensor *self_, float value)
{
  THAssert(THClTensor_checkGPU(state, 1, self_));
  if (!THClTensor_pointwiseApply1(state, self_, TensorFillOp(value))) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }
}

void THClTensor_zero(THClState *state, THClTensor *self_)
{
  THAssert(THClTensor_checkGPU(state, 1, self_));
//  if (THClTensor_isContiguous(state, self_)) {
//    THClCheck(cudaMemsetAsync(THClTensor_data(state, self_),
//                                0,
//                                sizeof(float) * THClTensor_nElement(state, self_),
//                                THClState_getCurrentStream(state)));
//  } else {
    if (!THClTensor_pointwiseApply1(state, self_, TensorFillOp(0))) {
      THArgCheck(false, 1, CLTORCH_DIM_WARNING);
    }
//  }
}

void THClTensor_zeros(THClState *state, THClTensor *r_, THLongStorage *size)
{
  THAssert(THClTensor_checkGPU(state, 1, r_));
  THClTensor_resize(state, r_, size, NULL);
  THClTensor_zero(state, r_);
//    THError("Not implemented");
}

void THClTensor_ones(THClState *state, THClTensor *r_, THLongStorage *size)
{
  THAssert(THClTensor_checkGPU(state, 1, r_));
  THClTensor_resize(state, r_, size, NULL);
  THClTensor_fill(state, r_, 1);
//    THError("Not implemented");
}

void THClTensor_reshape(THClState *state, THClTensor *r_, THClTensor *t, THLongStorage *size)
{
//  THAssert(THClTensor_checkGPU(state, 2, r_, t));
//  THClTensor_resize(state, r_, size, NULL);
//  THClTensor_copy(state, r_, t);
    THError("Not implemented");
//    return 0;
}

long THClTensor_numel(THClState *state, THClTensor *t)
{
  return THClTensor_nElement(state, t);
}

class TensorCPowOp : public HasOperator2, public HasOperator3 {
public:
  string operator2() const {
    return "*out = native_powr(*out, *in1)";
  }
  string operator3() const {
    return "*out = native_powr(*in1, *in2)";
  }
};

void THClTensor_cpow(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2));
  THArgCheck(THClTensor_nElement(state, src1) ==
             THClTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self = pow(self, src2)
    if (!THClTensor_pointwiseApply2(state, self_, src2, TensorCPowOp())) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src1);

    // self = pow(src1, src2)
    if (!THClTensor_pointwiseApply3(state, self_, src1, src2, TensorCPowOp())) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

class TensorDivOp : public HasOperator2, public HasOperator3 {
public:
  string operator2() const {
    return "*out /= *in1";
  }
  string operator3() const {
    return "*out = *in1 / *in2";
  }
};

void THClTensor_cdiv(THClState* state, THClTensor *self_, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2));
  THArgCheck(THClTensor_nElement(state, src1) ==
             THClTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self *= src2
    if (!THClTensor_pointwiseApply2(state, self_, src2, TensorDivOp())) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src1);

    // self = src1 * src2
    if (!THClTensor_pointwiseApply3(state, self_, src1, src2, TensorDivOp())) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

//struct TensorAddCMulOp {
//  TensorAddCMulOp(float v) : val(v) {}

//  __device__ __forceinline__ void
//  operator()(float* out, float* in1, float* in2) {
//    *out += val * *in1 * *in2;
//  }

//  float val;
//};

void THClTensor_addcmul(THClState *state, THClTensor *self_, THClTensor *t, float value, THClTensor *src1, THClTensor *src2)
{
//  THAssert(THClTensor_checkGPU(state, 4, self_, t, src1, src2));
//  if(self_ != t)
//  {
//    THClTensor_resizeAs(state, self_, t);
//    THClTensor_copy(state, self_, t);
//  }
//  THClTensor_resizeAs(state, self_, src1);

//  THArgCheck(THClTensor_nElement(state, src1) ==
//             THClTensor_nElement(state, src2), 3, "sizes do not match");

//  if (!THClTensor_pointwiseApply3(state, self_, src1, src2, TensorAddCMulOp(value))) {
//    THArgCheck(false, 2, CLTORCH_DIM_WARNING);
//  }

//  THClCheck(cudaGetLastError());
    THError("Not implemented");
}

//struct TensorAddCDivOp {
//  TensorAddCDivOp(float v) : val(v) {}

//  __device__ __forceinline__ void
//  operator()(float* out, float* in1, float* in2) {
//    *out += val * *in1 / *in2;
//  }

//  float val;
//};

void THClTensor_addcdiv(THClState *state, THClTensor *self_, THClTensor *t, float value, THClTensor *src1, THClTensor *src2)
{
//  THAssert(THClTensor_checkGPU(state, 4, self_, t, src1, src2));
//  if(self_ != t)
//  {
//    THClTensor_resizeAs(state, self_, t);
//    THClTensor_copy(state, self_, t);
//  }

//  THClTensor_resizeAs(state, self_, src1);
//  THArgCheck(THClTensor_nElement(state, src1) == THClTensor_nElement(state, src2), 3, "sizes do not match");

//  if (!THClTensor_pointwiseApply3(state, self_, src1, src2, TensorAddCDivOp(value))) {
//    THArgCheck(false, 2, CLTORCH_DIM_WARNING);
//  }

//  THClCheck(cudaGetLastError());
    THError("Not implemented");
//    return 0;
}

float THClTensor_minall(THClState *state, THClTensor *self)
{
//  THAssert(THClTensor_checkGPU(state, 1, self));
//  self = THClTensor_newContiguous(state, self);
//  thrust::device_ptr<float> self_data(THClTensor_data(state, self));

//  float result = thrust::reduce(self_data, self_data+THClTensor_nElement(state, self), (float)(THInf), thrust::minimum<float>());

//  THClTensor_free(state, self);
//  return result;
    THError("Not implemented");
    return 0;
}

float THClTensor_maxall(THClState *state, THClTensor *self)
{
  THAssert(THClTensor_checkGPU(state, 1, self));
  float val = (float) -THInf;
  CopyOp modifyOp;
  MaxOp reduceOp;
  if (!THClTensor_reduceAll(state, self,
          &modifyOp,
          &reduceOp,
          (float) -THInf, &val)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }

  return val;
}

float THClTensor_sumall(THClState *state, THClTensor *self)
{
  THAssert(THClTensor_checkGPU(state, 1, self));
  float val = 0.0f;
  CopyOp modifyOp;
  TensorAddOp reduceOp;
  if (!THClTensor_reduceAll(state, self,
          &modifyOp,
          &reduceOp,
          0.0f, &val)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }

  return val;
}

float THClTensor_prodall(THClState *state, THClTensor *self)
{
//  THAssert(THClTensor_checkGPU(state, 1, self));
//  self = THClTensor_newContiguous(state, self);
//  thrust::device_ptr<float> self_data(THClTensor_data(state, self));

//  float result = thrust::reduce(self_data, self_data+THClTensor_nElement(state, self), (float)(1), thrust::multiplies<float>());

//  THClTensor_free(state, self);
//  return result;
    THError("Not implemented");
    return 0;
}

//struct dim4 {
//    unsigned arr[4];

//    __host__ dim4(unsigned init=0) {
//        for(unsigned i=0; i<4; i++) { arr[i] = init; }
//    }

//    __host__ __device__ unsigned& operator[](const unsigned& idx) { return arr[idx]; }
//};

void THClTensor_sum(THClState* state, THClTensor *self, THClTensor *src, long dimension)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  CopyOp modifyOp;
  TensorAddOp reduceOp;
  THClTensor_reduceDim(
    state, self, src,
      0.0f, 
     &modifyOp, &reduceOp, dimension);


//  THError("Not implemented");
}

void THClTensor_prod(THClState* state, THClTensor *self, THClTensor *src, long dimension)
{
//  THAssert(THClTensor_checkGPU(state, 2, self, src));
//  THClTensor_reduceDim(
//    state, self, src,
//    thrust::identity<float>(), thrust::multiplies<float>(), 1.0f, dimension);

//  THClCheck(cudaGetLastError());
    THError("Not implemented");
//    return 0;
}

//struct logicalall_functor
//{
//  __host__ __device__ float operator()(const float& x, const float& y) const
//  {
//    return x && y;
//  }
//};

//struct logicalany_functor
//{
//  __host__ __device__ float operator()(const float& x, const float& y) const
//  {
//    return x || y;
//  }
//};

int THClTensor_logicalall(THClState *state, THClTensor *self) {
//  THAssert(THClTensor_checkGPU(state, 1, self));
//  self = THClTensor_newContiguous(state, self);
//  thrust::device_ptr<float> self_data(THClTensor_data(state, self));

//  int result = thrust::reduce(self_data, self_data+THClTensor_nElement(state, self), (float)(1), logicalall_functor());

//  THClTensor_free(state, self);
//  return result;
    THError("Not implemented");
    return 0;
}

int THClTensor_logicalany(THClState *state, THClTensor *self) {
//  THAssert(THClTensor_checkGPU(state, 1, self));
//  self = THClTensor_newContiguous(state, self);
//  thrust::device_ptr<float> self_data(THClTensor_data(state, self));

//  int result = thrust::reduce(self_data, self_data+THClTensor_nElement(state, self), (float)(0), logicalany_functor());

//  THClTensor_free(state, self);
//  return result;
    THError("Not implemented");
    return 0;
}
