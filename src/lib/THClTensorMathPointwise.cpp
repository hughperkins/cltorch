#include "THClTensorMath.h"
#include "THClGeneral.h"
//#include "THClBlas.h"
#include "THClTensorCopy.h"
//#include "THClTensorRandom.h"
#include "THClApply.h"
//#include "THClReduce.cuh"
#include "THClTensorMathPointwise.h"
#include "util/easycl_stringhelper.h"

#include <string>
using namespace std;

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#define IMPLEMENT_CL_TENSOR_BASIC_FUNC(NAME, CFUNC)                   \
  void THClTensor_##NAME(THClState* state, THClTensor* self_, THClTensor* src) { \
    THAssert(THClTensor_checkGPU(state, 2, self_, src));                \
    if (self_ == src) {                                                 \
      TensorGenOp op(CFUNC); \
      if (!THClTensor_pointwiseApply1(state, self_, &op)) { \
        THArgCheck(false, 2, CLTORCH_DIM_WARNING); \
      }                                                                 \
    } else {                                                            \
      THClTensor_resizeAs(state, self_, src);                         \
                                                                        \
      TensorGenOp op(CFUNC); \
      if (!THClTensor_pointwiseApply2(state, self_, src, &op)) { \
        THArgCheck(false, 2, CLTORCH_DIM_WARNING); \
      }                                                                 \
    }                                                                   \
                                                                        \
  }

#define MATHMODE "" // eg "native_" "half_"

IMPLEMENT_CL_TENSOR_BASIC_FUNC(log, MATHMODE "log")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(log1p, "log1p")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(exp, MATHMODE "exp")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(cos, MATHMODE "cos")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(acos, "acos")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(cosh, "cosh")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(sin, MATHMODE "sin")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(asin, "asin")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(sinh, "sinh")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(tan, MATHMODE "tan")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(pow, MATHMODE "powr")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(atan, "atan")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(tanh, "tanh")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(sqrt, MATHMODE "sqrt")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(ceil, "ceil")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(floor, "floor")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(abs, "fabs")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(round, "round")
IMPLEMENT_CL_TENSOR_BASIC_FUNC(neg, "-")

#undef IMPLEMENT_CL_TENSOR_BASIC_FUNC

void THClTensor_sigmoid(THClState* state, THClTensor* self_, THClTensor* src) {
  THAssert(THClTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    TensorSigmoidOp op;
    if (!THClTensor_pointwiseApply1(state, self_, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src);

    TensorSigmoidOp op;
    if (!THClTensor_pointwiseApply2(state, self_, src, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

void THClTensor_apply(THClState* state, THClTensor* self, char const *_operation1) {
//  THAssert(THClTensor_checkGPU(state, 2, self, src));
//  if (self == src) {
  string operation1 = easycl::replaceGlobal(string(_operation1), "exp", "ejp");
  operation1 = easycl::replaceGlobal(operation1, "max", "mam");
  operation1 = easycl::replaceGlobal(operation1, "x", "(*out)");
  operation1 = easycl::replaceGlobal(operation1, "ejp", "exp");
  operation1 = easycl::replaceGlobal(operation1, "mam", "max");
    TensorGenOpFullInline1 op(operation1);
    if (!THClTensor_pointwiseApply1(state, self, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
//  } else {
//    THError("not implemented (though we could... :-) )");
//    THClTensor_resizeAs(state, self_, src);

//    if (!THClTensor_pointwiseApply2(state, self_, src, TensorGenOp(#CFUNC))) {
//      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
//    }
//  }
}

void THClTensor_map(THClState* state, THClTensor* self, THClTensor *in1, char const *_operation2) {
  THAssert(THClTensor_checkGPU(state, 2, self, in1));
//  if (self == src) {
  string operation2 = easycl::replaceGlobal(string(_operation2), "exp", "ejp");
  operation2 = easycl::replaceGlobal(operation2, "max", "mam");
  operation2 = easycl::replaceGlobal(operation2, "x", "(*out)");
  operation2 = easycl::replaceGlobal(operation2, "ejp", "exp");
  operation2 = easycl::replaceGlobal(operation2, "mam", "max");
  operation2 = easycl::replaceGlobal(operation2, "y", "(*in1)");
    TensorGenOpFullInline2 op(operation2);
    if (!THClTensor_pointwiseApply2(state, self, in1, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
//  } else {
//    THError("not implemented (though we could... :-) )");
//    THClTensor_resizeAs(state, self_, src);

//    if (!THClTensor_pointwiseApply2(state, self_, src, TensorGenOp(#CFUNC))) {
//      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
//    }
//  }
}

void THClTensor_map2(THClState* state, THClTensor* self, THClTensor *in1, THClTensor *in2, char const *_operation3) {
  THAssert(THClTensor_checkGPU(state, 3, self, in1, in2));
  string operation3 = easycl::replaceGlobal(string(_operation3), "exp", "ejp");
  operation3 = easycl::replaceGlobal(operation3, "max", "mam");
  operation3 = easycl::replaceGlobal(operation3, "x", "(*out)");
  operation3 = easycl::replaceGlobal(operation3, "ejp", "exp");
  operation3 = easycl::replaceGlobal(operation3, "mam", "max");
  operation3 = easycl::replaceGlobal(operation3, "y", "(*in1)");
  operation3 = easycl::replaceGlobal(operation3, "z", "(*in2)");
//  if (self == src) {
    TensorGenOpFullInline3 op(operation3);
    if (!THClTensor_pointwiseApply3(state, self, in1, in2, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
//  } else {
//    THError("not implemented (though we could... :-) )");
//    THClTensor_resizeAs(state, self_, src);

//    if (!THClTensor_pointwiseApply2(state, self_, src, TensorGenOp(#CFUNC))) {
//      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
//    }
//  }
}

void THClTensor_cadd(THClState *state, THClTensor *self_, THClTensor* src1, float value, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2));
  if( src2->nDimension == 0 ) {
    THClTensor_add_gpu(state, self_, src1, src2);
    return;
  }
  THArgCheck(THClTensor_nElement(state, src1) ==
             THClTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    if (value == 1.0f) {
      // self += src2
      TensorAddOp op;
      if (!THClTensor_pointwiseApply2(state, self_, src2, &op)) {
        THArgCheck(false, 2, CLTORCH_DIM_WARNING);
      }
    } else {
      // self += value * src2
      TensorCAddOp op(value);
      if (!THClTensor_pointwiseApply2(state, self_, src2, &op)) {
        THArgCheck(false, 2, CLTORCH_DIM_WARNING);
      }
    }
  } else {
    THClTensor_resizeAs(state, self_, src1);

    if (value == 1.0f) {
      // self = src1 + src2
      TensorAddOp op;
      if (!THClTensor_pointwiseApply3(state, self_, src1, src2, &op)) {
        THArgCheck(false, 2, CLTORCH_DIM_WARNING);
      }
    } else {
      // self = src1 + value * src2
      TensorCAddOp op(value);
      if (!THClTensor_pointwiseApply3(state, self_, src1, src2, &op)) {
        THArgCheck(false, 2, CLTORCH_DIM_WARNING);
      }
    }
  }
}

class TensorMaxOp : public HasOperator2, public HasOperator3 {
public:
  string operator2() const {
    return "*out = max(*out, *in1)";
  }
  string operator3() const {
    return "*out = max(*in1, *in2)";
  }
};

void THClTensor_cmax(THClState *state, THClTensor *self, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self, src1, src2));
  THArgCheck(THClTensor_nElement(state, src1) ==
             THClTensor_nElement(state, src2), 2, "sizes do not match");

  TensorMaxOp op;
  if (self == src1) {
    if (!THClTensor_pointwiseApply2(state, self, src2, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self, src1);
    if (!THClTensor_pointwiseApply3(state, self, src1, src2, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

class TensorMinOp : public HasOperator2, public HasOperator3 {
public:
  string operator2() const {
    return "*out = min(*out, *in1)";
  }
  string operator3() const {
    return "*out = min(*in1, *in2)";
  }
};

void THClTensor_cmin(THClState *state, THClTensor *self, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self, src1, src2));
  THArgCheck(THClTensor_nElement(state, src1) ==
             THClTensor_nElement(state, src2), 2, "sizes do not match");

  TensorMinOp op;
  if (self == src1) {
    if (!THClTensor_pointwiseApply2(state, self, src2, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self, src1);
    if (!THClTensor_pointwiseApply3(state, self, src1, src2, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

class TensorMaxValueOp : public HasOperator1, public HasOperator2, public HasScalars {
public:
  int getNumScalars() const {
    return 1;
  }
  float getScalar(int index) const {
    return val;
  }
  TensorMaxValueOp(float v) : val(v) {}
  string operator1() const {
    return "*out = max(*out, val1)";
  }
  string operator2() const {
    return "*out = max(*in1, val1)";
  }

  float val;
};

void THClTensor_cmaxValue(THClState *state, THClTensor *self, THClTensor *src, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));

  TensorMaxValueOp op(value);
  if (self == src) {
    if (!THClTensor_pointwiseApply1(state, self, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self, src);
    if (!THClTensor_pointwiseApply2(state, self, src, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

class TensorMinValueOp : public HasOperator1, public HasOperator2, public HasScalars {
public:
  int getNumScalars() const {
    return 1;
  }
  float getScalar(int index) const {
    return val;
  }
  TensorMinValueOp(float v) : val(v) {}
  string operator1() const {
    return "*out = min(*out, val1)";
  }
  string operator2() const {
    return "*out = min(*in1, val1)";
  }

  float val;
};

void THClTensor_cminValue(THClState *state, THClTensor *self, THClTensor *src, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));

  TensorMinValueOp op(value);
  if (self == src) {
    if (!THClTensor_pointwiseApply1(state, self, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self, src);
    if (!THClTensor_pointwiseApply2(state, self, src, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

void THClTensor_csub(THClState *state, THClTensor *self_, THClTensor* src1, float value, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2));
  if(src2->nDimension == 0) {
    THClTensor_sub_gpu(state, self_, src1, src2);
    return;
  }
  THArgCheck(THClTensor_nElement(state, src1) ==
             THClTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    if (value == 1.0f) {
      // self += src2
      TensorSubOp op;
      if (!THClTensor_pointwiseApply2(state, self_, src2, &op)) {
        THArgCheck(false, 2, CLTORCH_DIM_WARNING);
      }
    } else {
      // self += value * src2
      TensorCSubOp op(value);
      if (!THClTensor_pointwiseApply2(state, self_, src2, &op)) {
        THArgCheck(false, 2, CLTORCH_DIM_WARNING);
      }
    }
  } else {
    THClTensor_resizeAs(state, self_, src1);

    if (value == 1.0f) {
      // self = src1 + src2
      TensorSubOp op;
      if (!THClTensor_pointwiseApply3(state, self_, src1, src2, &op)) {
        THArgCheck(false, 2, CLTORCH_DIM_WARNING);
      }
    } else {
      // self = src1 + value * src2
      TensorCSubOp op(value);
      if (!THClTensor_pointwiseApply3(state, self_, src1, src2, &op)) {
        THArgCheck(false, 2, CLTORCH_DIM_WARNING);
      }
    }
  }
}

void THClTensor_cmul(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2)
{
  THAssert(THClTensor_checkGPU(state, 3, self_, src1, src2));
  THArgCheck(THClTensor_nElement(state, src1) ==
             THClTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self *= src2
    TensorMulOp op;
    if (!THClTensor_pointwiseApply2(state, self_, src2, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  } else {
    THClTensor_resizeAs(state, self_, src1);

    // self = src1 * src2
    TensorMulOp op;
    if (!THClTensor_pointwiseApply3(state, self_, src1, src2, &op)) {
      THArgCheck(false, 2, CLTORCH_DIM_WARNING);
    }
  }
}

