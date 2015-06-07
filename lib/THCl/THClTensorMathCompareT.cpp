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

template<class Op>
void THClTensor_logicalTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2, Op op)
{
  THClTensor_resizeAs(state, self_, src1);
  THArgCheck(THClTensor_nElement(state, src1) == THClTensor_nElement(state, src2), 3, "sizes do not match");

  if (!THClTensor_pointwiseApply3(state, self_, src1, src2, op)) {
    THArgCheck(false, 2, CLNN_DIM_WARNING);
  }

//  THClCheck(cudaGetLastError());
}


// these should be generatable really...
struct TensorLTOp {
  string operator3() {
    return "*out = (float) (*in1 < *in2)";
  }
};
struct TensorGenLogOp {
  string logop;
  TensorGenLogOp(string logop) {
    this->logop = logop;
  }
  string operator3() {
    return "*out = (float) (*in1 " + logop + " *in2)";
  }
};
/*
struct TensorGTOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a > *b);
  }
};

struct TensorLEOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a <= *b);
  }
};

struct TensorGEOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a >= *b);
  }
};

struct TensorEQOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a == *b);
  }
};

struct TensorNEOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a != *b);
  }
};
*/
void THClTensor_ltTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2));
  THClTensor_logicalTensor(state, self_, src1, src2, TensorLTOp());
}

void THClTensor_gtTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2));
  THClTensor_logicalTensor(state, self_, src1, src2, TensorGenLogOp(">"));
}
/*


void THClTensor_leTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2));
  THClTensor_logicalTensor(state, self_, src1, src2, TensorLEOp());
}


void THClTensor_geTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2));
  THClTensor_logicalTensor(state, self_, src1, src2, TensorGEOp());
}


void THClTensor_eqTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2));
  THClTensor_logicalTensor(state, self_, src1, src2, TensorEQOp());
}


void THClTensor_neTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2));
  THClTensor_logicalTensor(state, self_, src1, src2, TensorNEOp());
}
*/

