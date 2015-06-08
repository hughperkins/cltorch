#include <string>

#include "THClTensorMath.h"
#include "THClGeneral.h"
//#include "THCBlas.h"
#include "THClTensorCopy.h"
//#include "THCTensorRandom.h"
#include "THClApply.h"
//#include "THCReduce.cuh"

using namespace std;

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

struct TensorAddConstantOp {
  bool has_scalar() { return true; }
  TensorAddConstantOp(float v) : val(v) {}
  string operator2() {
    return "*out = *in1 + val";
  }
  string operator1() {
    return "*out += val";
  }

//  __device__ __forceinline__ void operator()(float* out, float* in) {
//    *out = *in + val;
//  }

//  __device__ __forceinline__ void operator()(float* v) {
//    *v += val;
//  }

  const float val;
};

void THClTensor_add(THClState *state, THClTensor *self_, THClTensor *src_, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THClTensor_pointwiseApply1(state, self_, TensorAddConstantOp(value))) {
      THArgCheck(false, 2, CLNN_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src_);

    if (!THClTensor_pointwiseApply2(state, self_, src_, TensorAddConstantOp(value))) {
      THArgCheck(false, 2, CLNN_DIM_WARNING);
    }
  }

//  THClCheck(cudaGetLastError());
}

struct TensorMulConstantOp {
//  TensorMulConstantOp(float v) : val(v) {}
//  __device__ __forceinline__ void operator()(float* out, float* in) {
//    *out = *in * val;
//  }

//  __device__ __forceinline__ void operator()(float* v) {
//    *v *= val;
//  }

  const float val;
};
/*
void THClTensor_mul(THClState *state, THClTensor *self_, THClTensor *src_, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THClTensor_pointwiseApply1(state, self_, TensorMulConstantOp(value))) {
      THArgCheck(false, 2, CLNN_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src_);

    if (!THClTensor_pointwiseApply2(state, self_, src_, TensorMulConstantOp(value))) {
      THArgCheck(false, 2, CLNN_DIM_WARNING);
    }
  }

//  THClCheck(cudaGetLastError());
}

void THClTensor_div(THClState* state, THClTensor *self_, THClTensor *src_, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(value != 0.0f, 3, "divide by zero");

  if (self_ == src_) {
    if (!THClTensor_pointwiseApply1(state, self_, TensorMulConstantOp(1.0f / value))) {
      THArgCheck(false, 2, CLNN_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src_);

    if (!THClTensor_pointwiseApply2(state, self_, src_, TensorMulConstantOp(1.0f / value))) {
      THArgCheck(false, 2, CLNN_DIM_WARNING);
    }
  }

//  THClCheck(cudaGetLastError());
}
*/

