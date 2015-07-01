#include <string>

#include "THClTensorMath.h"
#include "THClGeneral.h"
//#include "THClBlas.h"
#include "THClTensorCopy.h"
//#include "THCTensorRandom.h"
#include "THClApply.h"
//#include "THCReduce.cuh"

using namespace std;

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

void THClTensor_logicalValue(THClState *state, THClTensor *self_, THClTensor *src, HasOperator2 *op)
{
  THClTensor_resizeAs(state, self_, src);

  if (!THClTensor_pointwiseApply2(state, self_, src, op)) {
    THArgCheck(false, 2, CLTORCH_DIM_WARNING);
  }
}

class TensorGenCompareValueOp : public HasOperator2, public HasScalars {
public:
  int getNumScalars() const { return 1; }
  float getScalar( int index ) const { return val; }
  TensorGenCompareValueOp(std::string op, float v) : 
    val(v),
    op(op) {}
  string operator2() const {
    return "*out = (*in1 " + op + " val1)";
  }
  const float val;
  std::string op;
};

#define GENERATE_THClTensor_LogValue(NAME, OP) \
 void THClTensor_##NAME##Value(THClState *state, THClTensor *self_, THClTensor *src, float value) \
{ \
  THAssert(THClTensor_checkGPU(state, 2, self_, src)); \
  TensorGenCompareValueOp op(#OP, value); \
  THClTensor_logicalValue(state, self_, src, &op); \
}

GENERATE_THClTensor_LogValue(ge, >=)
GENERATE_THClTensor_LogValue(ne, !=)
GENERATE_THClTensor_LogValue(eq, ==)
GENERATE_THClTensor_LogValue(le, <=)
GENERATE_THClTensor_LogValue(lt, <)
GENERATE_THClTensor_LogValue(gt, >)


