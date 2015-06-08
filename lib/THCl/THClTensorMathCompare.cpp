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
}

struct TensorGenCompareValueOp {
  bool has_scalar() { return true; }
  TensorGenCompareValueOp(std::string op, float v) : 
    val(v),
    op(op) {}
  string operator2() {
    return "*out = (*in1 " + op + " val)";
  }
  const float val;
  std::string op;
};

#define GENERATE_THClTensor_LogValue(NAME, OP) \
 void THClTensor_##NAME##Value(THClState *state, THClTensor *self_, THClTensor *src, float value) \
{ \
  THAssert(THClTensor_checkGPU(state, 2, self_, src)); \
  THClTensor_logicalValue(state, self_, src, TensorGenCompareValueOp(#OP, value)); \
}

GENERATE_THClTensor_LogValue(ge, >=)
GENERATE_THClTensor_LogValue(ne, !=)
GENERATE_THClTensor_LogValue(eq, ==)
GENERATE_THClTensor_LogValue(le, <=)
GENERATE_THClTensor_LogValue(lt, <)
GENERATE_THClTensor_LogValue(gt, >)


