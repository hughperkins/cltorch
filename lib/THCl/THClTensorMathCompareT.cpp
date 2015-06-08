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
}

struct TensorGenLogOp {
  bool has_scalar(){ return false; }
  float val;
  string logop;
  TensorGenLogOp(string logop) {
    this->logop = logop;
  }
  string operator3() {
    return "*out = (float) (*in1 " + logop + " *in2)";
  }
};

#define GENERATE_THClTensor_LogOpTensor(NAME, LOGOP) \
void THClTensor_##NAME##Tensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2)  \
{  \
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2)); \
  THClTensor_logicalTensor(state, self_, src1, src2, TensorGenLogOp(#LOGOP)); \
}

GENERATE_THClTensor_LogOpTensor(lt, <)
GENERATE_THClTensor_LogOpTensor(gt, >)
GENERATE_THClTensor_LogOpTensor(le, <=)
GENERATE_THClTensor_LogOpTensor(ge, >=)
GENERATE_THClTensor_LogOpTensor(ne, !=)
GENERATE_THClTensor_LogOpTensor(eq, ==)


