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

#define CATCH_EXCEPT(method) \
try { \
  method; \
} catch(exception &e) { \
  THError("Something went wrong: %s", e.what()); \
}

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

class TensorFillPointTensorOp : public HasOperator1, public HasPointTensors {
public:
  int getNumPointTensors() const { return 1; }
  const THClTensor *getPointTensor( int index ) const { return val; }
  TensorFillPointTensorOp(THClTensor *v) : val(v) {}
  string operator1() const {
    return "*out = *pointTensor1";
  }
  const THClTensor *val;
};

void THClTensor_fill(THClState* state, THClTensor *self_, float value)
{
  THAssert(THClTensor_checkGPU(state, 1, self_));
  TensorFillOp op(value);
  if (!THClTensor_pointwiseApply1(state, self_, &op)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }
}

void THClTensor_fill_gpu(THClState* state, THClTensor *self_, THClTensor *value)
{
  THAssert(THClTensor_checkGPU(state, 2, self_, value));
  TensorFillPointTensorOp op(value);
  if (!THClTensor_pointwiseApply1(state, self_, &op)) {
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
    TensorFillOp op(0);
    if (!THClTensor_pointwiseApply1(state, self_, &op)) {
      THArgCheck(false, 1, CLTORCH_DIM_WARNING);
    }
//  }
}

void THClTensor_zeros(THClState *state, THClTensor *r_, THLongStorage *size)
{
  THAssert(THClTensor_checkGPU(state, 1, r_));
  THClTensor_resize(state, r_, size, NULL);
  THClTensor_zero(state, r_);
}

void THClTensor_ones(THClState *state, THClTensor *r_, THLongStorage *size)
{
  THAssert(THClTensor_checkGPU(state, 1, r_));
  THClTensor_resize(state, r_, size, NULL);
  THClTensor_fill(state, r_, 1);
}

void THClTensor_reshape(THClState *state, THClTensor *r_, THClTensor *t, THLongStorage *size)
{
  THAssert(THClTensor_checkGPU(state, 2, r_, t));
  THClTensor_resize(state, r_, size, NULL);
  THClTensor_copy(state, r_, t);
//    THError("Not implemented");
//    return 0;
}

long THClTensor_numel(THClState *state, THClTensor *t)
{
  return THClTensor_nElement(state, t);
}

class TensorCPowOp : public HasOperator2, public HasOperator3 {
public:
  string operator2() const {
    return "*out = pow(*out, *in1)";
  }
  string operator3() const {
    return "*out = pow(*in1, *in2)";
  }
};

void THClTensor_cpow(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2));
  THArgCheck(THClTensor_nElement(state, src1) ==
             THClTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self = pow(self, src2)
    TensorCPowOp op;
    if (!THClTensor_pointwiseApply2(state, self_, src2, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src1);

    // self = pow(src1, src2)
    TensorCPowOp op;
    if (!THClTensor_pointwiseApply3(state, self_, src1, src2, &op)) {
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
    TensorDivOp op;
    if (!THClTensor_pointwiseApply2(state, self_, src2, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src1);

    // self = src1 * src2
    TensorDivOp op;
    if (!THClTensor_pointwiseApply3(state, self_, src1, src2, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

class TensorAddCMulOp : public HasScalars, public HasOperator3 {
public:
  TensorAddCMulOp(float v) : val(v) {}
  int getNumScalars() const{ return 1; }
  float getScalar(int index) const{ return val; }
  std::string operator3() const {
    return "*out += val1 * *in1 * *in2";
  }
  
//  __device__ __forceinline__ void
//  operator()(float* out, float* in1, float* in2) {
//    *out += val * *in1 * *in2;
//  }

  float val;
};

void THClTensor_addcmul(THClState *state, THClTensor *self_, THClTensor *t, float value, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 4, self_, t, src1, src2));
  if(self_ != t)
  {
    THClTensor_resizeAs(state, self_, t);
    THClTensor_copy(state, self_, t);
  }
  else
  {
    THArgCheck(THClTensor_nElement(state, self_) == THClTensor_nElement(state, src1),
               1, "sizes do not match");
  }

  THArgCheck(THClTensor_nElement(state, src1) == THClTensor_nElement(state, src2),
             3, "sizes do not match");

  TensorAddCMulOp op(value);
  if (!THClTensor_pointwiseApply3(state, self_, src1, src2, &op)) {
    THArgCheck(false, 2, CLTORCH_DIM_WARNING);
  }

}

class TensorAddCDivOp : public HasScalars, public HasOperator3 {
public:
  TensorAddCDivOp(float v) : val(v) {}

  int getNumScalars() const{ return 1; }
  float getScalar(int index) const{ return val; }
  std::string operator3() const {
    return "*out += val1 * *in1 / *in2";
  }
//  __device__ __forceinline__ void
//  operator()(float* out, float* in1, float* in2) {
//    *out += val * *in1 / *in2;
//  }

  float val;
};

void THClTensor_addcdiv(THClState *state, THClTensor *self_, THClTensor *t, float value, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 4, self_, t, src1, src2));
  if(self_ != t)
  {
    THClTensor_resizeAs(state, self_, t);
    THClTensor_copy(state, self_, t);
  }
  else
  {
    THArgCheck(THClTensor_nElement(state, self_) == THClTensor_nElement(state, src1),
               1, "sizes do not match");
  }
  THArgCheck(THClTensor_nElement(state, src1) == THClTensor_nElement(state, src2),
             3, "sizes do not match");

  TensorAddCDivOp op(value);
  if (!THClTensor_pointwiseApply3(state, self_, src1, src2, &op)) {
    THArgCheck(false, 2, CLTORCH_DIM_WARNING);
  }


}

void THClTensor_minall_gpu(THClState *state, THClTensor *self, THClTensor *src)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  CopyOp modifyOp;
  MinOp reduceOp;
  THClTensor_resize0d(state, self);
  if (!THClTensor_reduceAll(state, src,
          &modifyOp,
          &reduceOp,
          FLT_MAX, self->storage->wrapper)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }
}

void THClTensor_maxall_gpu(THClState *state, THClTensor *self, THClTensor *src)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  CopyOp modifyOp;
  MaxOp reduceOp;
  THClTensor_resize0d(state, self);
  if (!THClTensor_reduceAll(state, src,
          &modifyOp,
          &reduceOp,
          -FLT_MAX, self->storage->wrapper)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }
}

float THClTensor_minall(THClState *state, THClTensor *self)
{
  THAssert(THClTensor_checkGPU(state, 1, self));
  float val = FLT_MAX;
  CopyOp modifyOp;
  MinOp reduceOp;
  const int device = self->storage->device;
  THClScratchSpace *scratch = THClState_getDeviceScratchSpace(state, device, 0);
  if (!THClTensor_reduceAll(state, self,
          &modifyOp,
          &reduceOp,
          FLT_MAX, scratch->wrapper)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }
  StatefulTimer::timeCheck("minall before copytohost");
  scratch->wrapper->copyToHost();
  StatefulTimer::timeCheck("minall after copytohost");
  val = scratch->data[0];

  return val;
}

float THClTensor_maxall(THClState *state, THClTensor *self)
{
  THAssert(THClTensor_checkGPU(state, 1, self));
  float val = -FLT_MAX;
  CopyOp modifyOp;
  MaxOp reduceOp;
  const int device = self->storage->device;
  THClScratchSpace *scratch = THClState_getDeviceScratchSpace(state, device, 0);
  if (!THClTensor_reduceAll(state, self,
          &modifyOp,
          &reduceOp,
          -FLT_MAX, scratch->wrapper)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }
  StatefulTimer::timeCheck("maxall before copytohost");
  scratch->wrapper->copyToHost();
  StatefulTimer::timeCheck("maxall after copytohost");
  val = scratch->data[0];

  return val;
}

float THClTensor_as_float(THClState *state, THClTensor *self)
{
  THAssert(THClTensor_checkGPU(state, 1, self));
  THArgCheck(self->nDimension == 0, 1, "tensor must be point tensor, zero dimensions");
  THArgCheck(self->storage != 0, 1, "tensor must have allocated storage");
  float val = 0.0f;
//  const int device = self->storage->device;
  StatefulTimer::timeCheck("THClTensor_as_float before copytohost");
  self->storage->wrapper->copyToHost();
  StatefulTimer::timeCheck("THClTensor_as_float after copytohost");
  val = self->storage->data[self->storageOffset];

  return val;
}

float THClTensor_sumall(THClState *state, THClTensor *self)
{
  THAssert(THClTensor_checkGPU(state, 1, self));
  float val = 0.0f;
  CopyOp modifyOp;
  TensorAddOp reduceOp;
  const int device = self->storage->device;
  THClScratchSpace *scratch = THClState_getDeviceScratchSpace(state, device, 0);
//  CLWrapper *devOut = scratch->wrapper;
  if (!THClTensor_reduceAll(state, self,
          &modifyOp,
          &reduceOp,
          0.0f,
          scratch->wrapper)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }
  StatefulTimer::timeCheck("sumall before copytohost");
  scratch->wrapper->copyToHost();
  StatefulTimer::timeCheck("sumall after copytohost");
  val = scratch->data[0];

  return val;
}

void THClTensor_sumall_gpu(THClState *state, THClTensor *self, THClTensor *src)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  CopyOp modifyOp;
  TensorAddOp reduceOp;
  THClTensor_resize0d(state, self);
  if (!THClTensor_reduceAll(state, src,
          &modifyOp,
          &reduceOp,
          0.0f, self->storage->wrapper)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }
}

float THClTensor_prodall(THClState *state, THClTensor *self)
{
  THAssert(THClTensor_checkGPU(state, 1, self));
  float val = 0.0f;
  CopyOp modifyOp;
  TensorMulOp reduceOp;
  const int device = self->storage->device;
  THClScratchSpace *scratch = THClState_getDeviceScratchSpace(state, device, 0);
  if (!THClTensor_reduceAll(state, self,
          &modifyOp,
          &reduceOp,
          1.0f,
          scratch->wrapper)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }

  StatefulTimer::timeCheck("prodall before copytohost");
  scratch->wrapper->copyToHost();
  StatefulTimer::timeCheck("prodall after copytohost");
  val = scratch->data[0];

  return val;
}

void THClTensor_prodall_gpu(THClState *state, THClTensor *self, THClTensor *src)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  CopyOp modifyOp;
  TensorMulOp reduceOp;
  THClTensor_resize0d(state, self);
  if (!THClTensor_reduceAll(state, src,
          &modifyOp,
          &reduceOp,
          1.0f, self->storage->wrapper)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }
}

void THClTensor_sum(THClState* state, THClTensor *self, THClTensor *src, long dimension)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  CopyOp modifyOp;
  TensorAddOp reduceOp;
  CATCH_EXCEPT(THClTensor_reduceDim(
    state, self, src,
      0.0f, 
     &modifyOp, &reduceOp, dimension));
}

void THClTensor_prod(THClState* state, THClTensor *self, THClTensor *src, long dimension)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  CopyOp modifyOp;
  TensorMulOp reduceOp;
  CATCH_EXCEPT(THClTensor_reduceDim(
    state, self, src,
      1.0f, 
     &modifyOp, &reduceOp, dimension));
}

class logicalall_functor : public HasOperator3
{
public:
  std::string operator3() const {
    return "*out = (int)*in1 && (int)*in2";
  }
};

struct logicalany_functor : public HasOperator3
{
public:
  std::string operator3() const {
    return "*out = (int)*in1 || (int)*in2";
  }
};

int THClTensor_logicalall(THClState *state, THClTensor *self) {
  THAssert(THClTensor_checkGPU(state, 1, self));
  float val = 0.0f;
  CopyOp modifyOp;
  logicalall_functor reduceOp;
  const int device = self->storage->device;
  THClScratchSpace *scratch = THClState_getDeviceScratchSpace(state, device, 0);
  if (!THClTensor_reduceAll(state, self,
          &modifyOp,
          &reduceOp,
          1.0f, scratch->wrapper)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }
  StatefulTimer::timeCheck("logicalall before copytohost");
  scratch->wrapper->copyToHost();
  StatefulTimer::timeCheck("logicalall after copytohost");
  val = scratch->data[0];

  return val;
}

void THClTensor_logicalall_gpu(THClState *state, THClTensor *self, THClTensor *src)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  CopyOp modifyOp;
  logicalall_functor reduceOp;
  THClTensor_resize0d(state, self);
  if (!THClTensor_reduceAll(state, src,
          &modifyOp,
          &reduceOp,
          1.0f, self->storage->wrapper)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }
}

void THClTensor_logicalany_gpu(THClState *state, THClTensor *self, THClTensor *src)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  CopyOp modifyOp;
  logicalany_functor reduceOp;
  THClTensor_resize0d(state, self);
  if (!THClTensor_reduceAll(state, src,
          &modifyOp,
          &reduceOp,
          0.0f, self->storage->wrapper)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }
}

int THClTensor_logicalany(THClState *state, THClTensor *self) {
  THAssert(THClTensor_checkGPU(state, 1, self));
  float val = 0.0f;
  CopyOp modifyOp;
  logicalany_functor reduceOp;
  const int device = self->storage->device;
  THClScratchSpace *scratch = THClState_getDeviceScratchSpace(state, device, 0);
  if (!THClTensor_reduceAll(state, self,
          &modifyOp,
          &reduceOp,
          0.0f, scratch->wrapper)) {
    THArgCheck(false, 1, CLTORCH_DIM_WARNING);
  }
  StatefulTimer::timeCheck("logicalany before copytohost");
  scratch->wrapper->copyToHost();
  StatefulTimer::timeCheck("logicalany after copytohost");
  val = scratch->data[0];

  return val;
}
