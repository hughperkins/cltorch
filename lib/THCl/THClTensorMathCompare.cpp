#include <string>

#include "THClTensorMath.h"
#include "THClGeneral.h"
//#include "THClBlas.h"
#include "THClTensorCopy.h"
//#include "THCTensorRandom.h"
#include "THClApply.h"
//#include "THCReduce.cuh"

//#include <thrust/device_ptr.h>
//#include <thrust/fill.h>
//#include <thrust/functional.h>
//#include <thrust/reduce.h>
//#include <thrust/inner_product.h>

using namespace std;

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

template<class Op>
void THClTensor_logicalValue(THClState *state, THClTensor *self_, THClTensor *src, Op op)
{
  THClTensor_resizeAs(state, self_, src);

  if (!THClTensor_pointwiseApply2(state, self_, src, op)) {
    THArgCheck(false, 2, CLNN_DIM_WARNING);
  }

//  THClCheck(cudaGetLastError());
}

struct TensorLTValueOp {
  TensorLTValueOp(float v) : value(v) {}
  string operator2() {
    return "*out = (*in < value)";
  }
//  __device__ __forceinline__ void operator()(float* out, float* in) {
//    *out = (*in < value);
//  }

  const float value;
};

void THClTensor_ltValue(THClState *state, THClTensor *self_, THClTensor *src, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src));
  THClTensor_logicalValue(state, self_, src, TensorLTValueOp(value));
}

struct TensorGTValueOp {
  TensorGTValueOp(float v) : value(v) {}
  string operator2() {
    return "*out = (*in > value)";
  }
//  __device__ __forceinline__ void operator()(float* out, float* in) {
//    *out = (*in > value);
//  }

  const float value;
};

void THClTensor_gtValue(THClState *state, THClTensor *self_, THClTensor *src, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src));
  THClTensor_logicalValue(state, self_, src, TensorGTValueOp(value));
}
/*
struct TensorLEValueOp {
  TensorLEValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in <= value);
  }

  const float value;
};

void THClTensor_leValue(THClState *state, THClTensor *self_, THClTensor *src, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src));
  THClTensor_logicalValue(state, self_, src, TensorLEValueOp(value));
}

struct TensorGEValueOp {
  TensorGEValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in >= value);
  }

  const float value;
};

void THClTensor_geValue(THClState *state, THClTensor *self_, THClTensor *src, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src));
  THClTensor_logicalValue(state, self_, src, TensorGEValueOp(value));
}

struct TensorEQValueOp {
  TensorEQValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in == value);
  }

  const float value;
};

void THClTensor_eqValue(THClState *state, THClTensor *self_, THClTensor *src, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src));
  THClTensor_logicalValue(state, self_, src, TensorEQValueOp(value));
}

struct TensorNEValueOp {
  TensorNEValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in != value);
  }

  const float value;
};

void THClTensor_neValue(THClState *state, THClTensor *self_, THClTensor *src, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src));
  THClTensor_logicalValue(state, self_, src, TensorNEValueOp(value));
}
*/
