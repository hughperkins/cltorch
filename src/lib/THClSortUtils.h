//// from lib/THC/THCSortUtils.cuh:

#ifndef THCL_SORT_UTILS_INC
#define THCL_SORT_UTILS_INC

//#include "THClReduceApplyUtils.cuh"

class SortUtilsComp {
public:
  virtual const char *getOperator() const = 0;  // eg "<" or ">"
};

class SortUtilsCompGT : public SortUtilsComp {
public:
  virtual const char *getOperator() const {
    return ">";
  }
};

class SortUtilsCompLT : public SortUtilsComp {
public:
  virtual const char *getOperator() const {
    return "<";
  }
};

template< typename IndexType >
void kernelLaunch_bitonicSortKVInPlace(
    THClState *state,
    dim3 grid, dim3 block,
    int KeyDims,
    int ValueDims,
    int Power2SortSize,
    TensorInfo<IndexType> *keys,
    IndexType keySlices,
    IndexType keySliceSize,
    IndexType keySliceStride,
    TensorInfo<IndexType> *values,
    IndexType valueSliceStride,
    SortUtilsComp *comp);

//std::string THClSortUtils_getKernelTemplate();

#endif // THCL_SORT_UTILS_INC

