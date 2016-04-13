#pragma once

//// from lib/THC/THCSortUtils.cuh:

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
void THClSortUtils_kernelLaunch_bitonicSortKVInPlace(
    THClState *state,
    dim3 grid, dim3 block,
    int KeyDims,
    int ValueDims,
    int Power2SortSize,
    const TensorInfo<IndexType> &keys,
    IndexType keySlices,
    IndexType keySliceSize,
    IndexType keySliceStride,
    const TensorInfo<IndexType> &values,
    IndexType valueSliceStride,
    SortUtilsComp *comp);

