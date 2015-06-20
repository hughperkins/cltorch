// from lib/THC/THCTensorMasked.cu:

#include "THClTensorMath.h"
#include "THClGeneral.h"
#include "THClBlas.h"
#include "THClTensorCopy.h"
//#include "THClTensorRandom.h"
#include "THClApply.h"
#include "THClReduce.h"

// The largest consecutive integer representable in float32 (2^24)
#define FLOAT32_MAX_CONSECUTIVE_INT 16777216.0f

class TensorMaskedFillOp : public HasOperator2, public HasScalars {
public:
  int getNumScalars() const { return 1; }
  float getScalar( int index ) const { return value; }
  TensorMaskedFillOp(float v) : value(v) {}
  std::string operator2() const {
    return "if( *in1 != 0.0f ) { *out = val1; }";
  }
  float value;
};

//struct TensorMaskedCopyOp {
//  TensorMaskedCopyOp(float* s, float* bm, float* ps)
//      : src(s),
//        baseMask(bm),
//        maskPrefixSum(ps) {
//  }

//  /*__device__*/ /*__forceline__*/ void operator()(float* out, float* mask) {
//    // Really mask should be `0` or `1` but we can't propagate errors here.
//    if (*mask != 0.0f) {
//      // We've already checked that this offset is <= 2^24, so this is ok.
//      int srcOffset = (int) (mask - baseMask);
//      *out = src[(int) maskPrefixSum[srcOffset]];
//    }
//  }

//  // Where we are copying from
//  float* src;

//  // The base address of mask so we can calculate offset
//  float* baseMask;

//  // The index we are copying from
//  float* maskPrefixSum;
//};

//class TensorMaskedSelectOp : public HasOperator3, public HasScalars {
//public:
//  int getNumScalars() const { return 1; }
//  string operator3() const {
//    return "if(*out != 0.0f){out[(int)*in1] = *in2; }";
//  }
//  TensorMaskedSelectOp(float* t) : out(t) {}
//  void operator()(float* mask, float* maskPrefixSum, float* in) {
//    // Really mask should be `0` or `1` but we can't propagate errors here.
//    if (*mask != 0.0f) {
//      out[(int) *maskPrefixSum] = *in;
//    }
//  }

//  float* out;
//};

void THClTensor_maskedFill(THClState* state,
                             THClTensor *tensor, THClTensor *mask, float value)
{
  THAssert(THClTensor_checkGPU(state, 2, tensor, mask));
  THArgCheck(THClTensor_nElement(state, tensor) ==
             THClTensor_nElement(state, mask),
             2, "sizes do not match");

  if (!THClTensor_pointwiseApply2(state, tensor, mask, TensorMaskedFillOp(value))) {
    THArgCheck(false, 2, CLTORCH_DIM_WARNING);
  }
}


//void THClTensor_maskedCopy(THClState* state,
//                             THClTensor *tensor, THClTensor *mask, THClTensor *src)
//{
//  THAssert(THClTensor_checkGPU(state, 3, tensor, src, mask));
//  long maskSize = THClTensor_nElement(state, mask);
//  long tensorSize = THClTensor_nElement(state, tensor);
//  long srcSize = THClTensor_nElement(state, src);

//  // Since we are performing a prefix sum of mask, it cannot exceed
//  // the size allowed in consecutive integers in float32
//  THArgCheck(maskSize <= (long) FLOAT32_MAX_CONSECUTIVE_INT,
//             3, "mask nElements exceeds single-precision float "
//             "consecutive integer precision size (2^24)");

//  // `mask` and `tensor` must have the same number of elements
//  THArgCheck(maskSize == tensorSize, 2,
//             "mask and tensor must have the same number of elements");

//  THClTensor* contigMask = THClTensor_newContiguous(state, mask);
//  long oneElements = (long) THClTensor_sumall(state, contigMask);

//  // The number of `1` elements present in the mask must be <= the
//  // number of elements available in `src`
//  if (oneElements > srcSize) {
//    THClTensor_free(state, contigMask);
//    THArgCheck(false, 2, "source nElements must be == mask `1` elements");
//  }

//  // Use a prefix sum to determine the copy locations of the masked elements
//  THClTensor* maskPrefixSum = THClTensor_new(state);
//  THClTensor_resizeAs(state, maskPrefixSum, contigMask);

//  // We are getting elements from `src` based on an offset from
//  // `maskPrefixSum`, so that should be made contiguous too
//  THClTensor* contigSrc = THClTensor_newContiguous(state, src);

////   thrust::device_ptr<float>
//    maskData(THClTensor_data(state, contigMask));
////   thrust::device_ptr<float>
//    maskPrefixSumData(THClTensor_data(state, maskPrefixSum));
////   thrust::exclusive_scan(maskData,
//                         maskData + THClTensor_nElement(state, contigMask),
//                         maskPrefixSumData);

//  // update `tensor` where `mask` == 1 but pull from `src` at
//  // maskPrefixSum
//  bool status = THClTensor_pointwiseApply2(
//    state, tensor, contigMask,
//    TensorMaskedCopyOp(THClTensor_data(state, contigSrc),
//                       THClTensor_data(state, contigMask),
//                       THClTensor_data(state, maskPrefixSum)));

//  THClTensor_free(state, contigSrc);
//  THClTensor_free(state, maskPrefixSum);
//  THClTensor_free(state, contigMask);

//  THArgCheck(status, 2, CLTORCH_DIM_WARNING);

//  THError("Not implemented");
//}

//void THClTensor_maskedSelect(THClState* state,
//                               THClTensor *tensor, THClTensor *src, THClTensor *mask)
//{
//  THAssert(THClTensor_checkGPU(state, 3, tensor, src, mask));
//  THArgCheck(THClTensor_nElement(state, mask) == THClTensor_nElement(state, src),
//             2, "sizes do not match");

//  // Since we are performing a prefix sum of mask, it cannot exceed
//  // the size allowed in consecutive integers in float32
//  THArgCheck(THClTensor_nElement(state, mask) <=
//             (long) FLOAT32_MAX_CONSECUTIVE_INT,
//             3, "mask nElements exceeds single-precision float "
//             "consecutive integer precision size (2^24)");

//  // Determine our output size
//  THClTensor* contigMask = THClTensor_newContiguous(state, mask);
//  long totalElements = (long) THClTensor_sumall(state, contigMask);

//  // This should be contiguous already, so no need to make it contig
//  // for the apply kernel
//  THClTensor_resize1d(state, tensor, totalElements);

//  // Use a prefix sum to determine the output locations of the masked elements
//  THClTensor* maskPrefixSum = THClTensor_new(state);
//  THClTensor_resizeAs(state, maskPrefixSum, contigMask);

////   thrust::device_ptr<float>
//    maskData(THClTensor_data(state, contigMask));
////   thrust::device_ptr<float>
//    maskPrefixSumData(THClTensor_data(state, maskPrefixSum));
////   thrust::exclusive_scan(maskData,
//                         maskData + THClTensor_nElement(state, contigMask),
//                         maskPrefixSumData);

//  // Then copy over the masked elements at their desired output index
//  bool status = THClTensor_pointwiseApply3(
//    state, contigMask, maskPrefixSum,
//    src, TensorMaskedSelectOp(THClTensor_data(state, tensor)));

//  THClTensor_free(state, contigMask);
//  THClTensor_free(state, maskPrefixSum);

//  THArgCheck(status, 2, CLTORCH_DIM_WARNING);

//  THError("Not implemented");
//}

//void THClTensor_maskedFillByte(THClState* state, THClTensor *tensor, THByteTensor *mask, float value)
//{
//  THAssert(THClTensor_checkGPU(state, 1, tensor));
//  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
//  THClTensor* maskCl = THClTensor_newWithSize(state, maskSize, NULL);
//  THLongStorage_free(maskSize);
//  THClTensor_copyByte(state, maskCl, mask);
//  THClTensor_maskedFill(state, tensor, maskCl, value);
//  THClTensor_free(state, maskCl);
//}

//void THClTensor_maskedCopyByte(THClState* state, THClTensor *tensor, THByteTensor *mask, THClTensor *src)
//{
//  THAssert(THClTensor_checkGPU(state, 2, tensor, src));
//  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
//  THClTensor* maskCl = THClTensor_newWithSize(state, maskSize, NULL);
//  THLongStorage_free(maskSize);
//  THClTensor_copyByte(state, maskCl, mask);
//  THClTensor_maskedCopy(state, tensor, maskCl, src);
//  THClTensor_free(state, maskCl);
//}

//void THClTensor_maskedSelectByte(THClState* state, THClTensor *tensor, THClTensor *src, THByteTensor *mask)
//{
//  THAssert(THClTensor_checkGPU(state, 2, tensor, src));
//  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
//  THClTensor* maskCl = THClTensor_newWithSize(state, maskSize, NULL);
//  THLongStorage_free(maskSize);
//  THClTensor_copyByte(state, maskCl, mask);
//  THClTensor_maskedSelect(state, tensor, src, maskCl);
//  THClTensor_free(state, maskCl);
//}

