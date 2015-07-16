#include <string>

#include "THClTensorMath.h"
#include "THClGeneral.h"
//#include "THCBlas.h"
#include "THClTensorCopy.h"
//#include "THCTensorRandom.h"
#include "THClApply.h"
//#include "THCReduce.cuh"
#include "THClTensorMathCompare.h"

using namespace std;

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

void THClTensor_logicalTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2, HasOperator3 *op)
{
  THClTensor_resizeAs(state, self_, src1);
  THArgCheck(THClTensor_nElement(state, src1) == THClTensor_nElement(state, src2), 3, "sizes do not match");

  if (!THClTensor_pointwiseApply3(state, self_, src1, src2, op)) {
    THArgCheck(false, 2, CLTORCH_DIM_WARNING);
  }
}

class TensorGenLogOp : public HasOperator3 {
public:
  string logop;
  TensorGenLogOp(string logop) {
    this->logop = logop;
  }
  string operator3() const {
    return "*out = (float) (*in1 " + logop + " *in2)";
  }
};

#define GENERATE_THClTensor_LogOpTensor(NAME, LOGOP) \
void THClTensor_##NAME##Tensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2)  \
{  \
  if( src2->nDimension == 0 ) { \
    THClTensor_##NAME##PointTensor(state, self_, src1, src2); \
    return; \
  } \
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2)); \
  TensorGenLogOp op(#LOGOP); \
  THClTensor_logicalTensor(state, self_, src1, src2, &op); \
}

GENERATE_THClTensor_LogOpTensor(lt, <)
GENERATE_THClTensor_LogOpTensor(gt, >)
GENERATE_THClTensor_LogOpTensor(le, <=)
GENERATE_THClTensor_LogOpTensor(ge, >=)
GENERATE_THClTensor_LogOpTensor(ne, !=)
GENERATE_THClTensor_LogOpTensor(eq, ==)


