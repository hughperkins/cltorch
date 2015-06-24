#include "THClTensorMath.h"
#include "THClGeneral.h"
//#include "THCBlas.h"
#include "THClTensorCopy.h"
//#include "THCTensorRandom.h"
#include "THClApply.h"
//#include "THCReduce.cuh"

#include <iostream>
#include <string>

using namespace std;

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

class TensorAddConstantOp : public HasOperator1, public HasOperator2, public HasScalars {
public:
  int getNumScalars() const { return 1; }
  float getScalar( int index ) const { return val; }
  TensorAddConstantOp(float v) : val(v) {}
  string operator2() const {
    return "*out = *in1 + val1";
  }
  string operator1() const {
    return "*out += val1";
  }
  const float val;
};

void THClTensor_add(THClState *state, THClTensor *self_, THClTensor *src_, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THClTensor_pointwiseApply1(state, self_, TensorAddConstantOp(value))) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src_);

    if (!THClTensor_pointwiseApply2(state, self_, src_, TensorAddConstantOp(value))) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

class TensorMulConstantOp : public HasOperator2, public HasOperator1, public HasScalars {
public:
  int getNumScalars() const { return 1; }
  float getScalar( int index ) const { return val; }
  TensorMulConstantOp(float v) : val(v) {}
  string operator2() const {
    return "*out = *in1 * val1";
  }
  string operator1() const {
    return "*out *= val1";
  }
  const float val;
};
void THClTensor_mul(THClState *state, THClTensor *self_, THClTensor *src_, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THClTensor_pointwiseApply1(state, self_, TensorMulConstantOp(value))) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src_);

    if (!THClTensor_pointwiseApply2(state, self_, src_, TensorMulConstantOp(value))) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

void THClTensor_div(THClState* state, THClTensor *self_, THClTensor *src_, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(value != 0.0f, 3, "divide by zero");

  if (self_ == src_) {
    if (!THClTensor_pointwiseApply1(state, self_, TensorMulConstantOp(1.0f / value))) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src_);

    if (!THClTensor_pointwiseApply2(state, self_, src_, TensorMulConstantOp(1.0f / value))) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}


