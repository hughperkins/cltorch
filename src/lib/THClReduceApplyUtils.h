#ifndef THCL_REDUCE_APPLY_UTILS_INC
#define THCL_REDUCE_APPLY_UTILS_INC

#include <string>
#include <assert.h>
#include <stdexcept>

#include "THGeneral.h"
#include "THClGeneral.h"
#include "THClTensor.h"
#include "THClOperators.h"
#include "util/easycl_stringhelper.h"

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

// Maximum number of dimensions allowed for cltorch
#define MAX_CLTORCH_DIMS 25

// Warning string for tensor arguments that are too large or have too
// many dimensions
#define CLTORCH_STR(X) #X
#define CLTORCH_DIM_WARNING "tensor too large or too many (>" \
  CLTORCH_STR(MAX_CLTORCH_DIMS) ") dimensions"

std::string THClReduceApplyUtils_getKernelTemplate();

// Enum that indicates whether tensor arguments are read/write or
// read-only
enum TensorArgType { ReadWrite, ReadOnly };
class CLWrapper;

// Copy operator for the pointwise apply kernel
class CopyOp : public HasOperator2 {
public:
    std::string operator2() const {
        return "*out = *in1";
    }
};

// CL kernel argument that defines tensor layout
template <typename IndexType>
struct TensorInfo {
  // Extracts size/stride information for the kernel.
  // Successive dimensions can be collapsed if the size/strides match
  // up and thus there are no holes between the dimensions. This is used
  // to reduce the complexity of the problem.
  // The optional `reduceDim` indicates a reduction dimension for the
  // given tensor, so that the output size for this dimension will be 1.
  TensorInfo(THClState* state, THClTensor* t, int reduceDim = -1);

  // Collapses all runs of successive dimensions if the size/strides
  // match up within the run and there are no holes between the
  // dimensions.
  // If excludeDim is set (not -1), then excludeDim will not be
  // collapsed with any other dimension.
  // Function returns the new dimension index that excludeDim maps to,
  // since the collapsed dimensions are <= the input dimensions.
  int collapseDims(int excludeDim = -1);

  // Contiguous tensors of more than one dimension are collapsed down
  // to one tensor
  inline bool isContiguous() const {
    return (dims == 1 && strides[0] == 1);
  }

  CLWrapper *wrapper;
  long offset;
//  float* data;
  IndexType sizes[MAX_CLTORCH_DIMS];
  IndexType strides[MAX_CLTORCH_DIMS];
  int dims;
};

template <typename IndexType>
TensorInfo<IndexType>::TensorInfo(THClState* state,
                                  THClTensor* t,
                                  int reduceDim)
    : wrapper(NULL), offset(0), dims(0) {

  offset = THClTensor_storageOffset(state, t);
  wrapper = THClTensor_wrapper(state, t);

  dims = THClTensor_nDimension(state, t);
  assert(dims <= MAX_CLTORCH_DIMS);

  for (int i = 0; i < dims; ++i) {
    sizes[i] = THClTensor_size(state, t, i);
    strides[i] = THClTensor_stride(state, t, i);
  }

  assert(reduceDim == -1 || (reduceDim < dims && reduceDim >= 0));

  if (reduceDim != -1) {
    sizes[reduceDim] = 1;
  }
}

template <typename IndexType>
int
TensorInfo<IndexType>::collapseDims(int excludeDim) {
  // Find the innermost dimension not of size 1, since dimensions of size 1 are
  // collapsible.
  int firstNonOneDim = -1;

  for (int i = dims - 1; i >= 0; --i) {
    if (i == excludeDim) {
      // We cannot collapse this dimension, even if it is size 1
      firstNonOneDim = i;
      break;
    }

    if (sizes[i] != 1) {
      firstNonOneDim = i;
      break;
    }
  }

  // Special case: if all dimensions are of size 1, then this is a
  // single-point tensor that we still have to operate on. Reduce to a
  // single point.
  if (firstNonOneDim == -1) {
    assert(excludeDim == -1);

    dims = 1;
    sizes[0] = 1;
    strides[0] = 1;

    // Everything effectively got collapsed into this dimension
    return 0;
  }

  // Count the number of successive dimensions that can be collapsed, from
  // innermost to outermost.
  int numCollapsed = 0;

  // Skip the leading size 1 dims
  numCollapsed += dims - 1 - firstNonOneDim;

  // We perform one pass through to determine how many dimensions we
  // can collapse, before calculating the actual size of the collapsed
  // dimensions.
  // size/strideInner are the size/strides of the previous inner
  // non-collapsible dim we encounter.
  long sizeInner = sizes[firstNonOneDim];
  long strideInner = strides[firstNonOneDim];

  for (int i = firstNonOneDim - 1; i >= 0; --i) {
    long sizeOuter = sizes[i];
    long strideOuter = strides[i];

    // Don't collapse this dimension if we want to exclude it from
    // collapsing.
    // Since this code is attempting to collapse a subsequent
    // dimension (i) with the preceding dimension (i + 1), we can only
    // perform collapsing if the preceding dimension can be collapsed
    // (i.e., not excludeDim)
    if ((excludeDim != i) && (excludeDim != i + 1)) {
      // The next outermost dimension can be skipped if size 1
      if (sizeOuter == 1) {
        ++numCollapsed;
        continue;
      }

      // If the next outermost dimension is contiguous with the
      // previous non-collapsed one, collapse it
      if (strideOuter == strideInner * sizeInner) {
        ++numCollapsed;

        // This is the run of collapsed dimensions' size
        sizeInner = sizeInner * sizeOuter;
        continue;
      }
    }

    // Otherwise, this new outer dimension at `i` cannot be collapsed
    // because it is excluded from collapsing, or it is not contiguous
    // with the previous inner dimension.
    sizeInner = sizeOuter;
    strideInner = strideOuter;
  }

  // This will be our new size/stride and dimension.
  IndexType newSizes[MAX_CLTORCH_DIMS];
  IndexType newStrides[MAX_CLTORCH_DIMS];

  assert(numCollapsed < dims);
  int newDims = dims - numCollapsed;

  // We return the index of the excluded dimension that is excluded
  // from being collapsed here.
  int returnDim = -1;

  // We perform a second pass through the dimensions to actually
  // calculate the size of the collapsed dimensions.
  int collapsedIndex = dims - numCollapsed - 1;
  newSizes[collapsedIndex] = sizes[firstNonOneDim];
  newStrides[collapsedIndex] = strides[firstNonOneDim];

  if (firstNonOneDim == excludeDim) {
    returnDim = collapsedIndex;
  }

  for (int i = firstNonOneDim - 1; i >= 0; --i) {
    IndexType sizeOuter = sizes[i];
    IndexType strideOuter = strides[i];

    if ((excludeDim != i) && (excludeDim != i + 1)) {
      if (sizeOuter == 1) {
        // skip
        continue;
      }

      if (strideOuter == newSizes[collapsedIndex] * newStrides[collapsedIndex]) {
        // collapse
        newSizes[collapsedIndex] *= sizeOuter;
        continue;
      }
    }

    // Otherwise, strides don't match, or dim `i` is excluded from
    // collapsing.
    --collapsedIndex;
    assert(collapsedIndex >= 0);
    assert(collapsedIndex < newDims);
    newSizes[collapsedIndex] = sizeOuter;
    newStrides[collapsedIndex] = strideOuter;

    if (excludeDim == i) {
      returnDim = collapsedIndex;
    }
  }

  // We must have filled all the dimensions we're looking for
  assert(collapsedIndex == 0);
  assert((excludeDim == -1) || (returnDim != -1));

  dims = newDims;

  for (int i = 0; i < dims; ++i) {
    sizes[i] = newSizes[i];
    strides[i] = newStrides[i];
  }

  // After collapsing, the original `excludeDim` may have been
  // renumbered to this new `returnDim`, since some dimensions could
  // have been collapsed.
  return returnDim;
}

typedef struct TensorInfoCl {
  TensorInfoCl( TensorInfo<unsigned int> info ) {
    dims = info.dims;
    if( info.offset > ( 1l << 30 ) ) {
      throw std::runtime_error("size " + easycl::toString(info.offset) + " out of bounds");
    }
    offset = (int)info.offset;
    for( int i = 0; i < dims; i++ ) {
      sizes[i] = info.sizes[i];
      strides[i] = info.strides[i];
    }
  }
  TensorInfoCl( TensorInfo<unsigned long> info ) {
    dims = info.dims;
    if( info.offset > ( 1l << 30 ) ) {
      throw std::runtime_error("size " + easycl::toString(info.offset) + " out of bounds");
    }
    offset = (int)info.offset;
    for( int i = 0; i < dims; i++ ) {
      if( info.sizes[i] > ( 1l << 31 ) ) {
        throw std::runtime_error("size " + easycl::toString(info.sizes[i]) + " out of bounds");
      }
      sizes[i] = info.sizes[i];
      strides[i] = info.strides[i];
    }
  }
  TensorInfoCl( TensorInfo<unsigned long long> info ) {
    dims = info.dims;
    if( info.offset > ( 1l << 30 ) ) {
      throw std::runtime_error("size " + easycl::toString(info.offset) + " out of bounds");
    }
    offset = (int)info.offset;
    for( int i = 0; i < dims; i++ ) {
      if( info.sizes[i] > ( 1l << 31 ) ) {
        throw std::runtime_error("size " + easycl::toString(info.sizes[i]) + " out of bounds");
      }
      sizes[i] = info.sizes[i];
      strides[i] = info.strides[i];
    }
  }
  TensorInfoCl(THClTensor *tensor ) {
    dims = tensor->nDimension;
    for( int i = 0; i < dims; i++ ) {
      sizes[i] = tensor->size[i];
      strides[i] = tensor->stride[i];
    }
    offset = tensor->storageOffset;
  }
  unsigned int sizes[MAX_CLTORCH_DIMS];
  unsigned int strides[MAX_CLTORCH_DIMS];
  int offset;
  int dims;
} TensorInfoCl;


// Translate a linear index for the apply to a float* offset;
// specialized on `Dims` to reduce nvcc compilation time
//template <typename IndexType, int Dims>
//struct IndexToOffset {
//  static __host__ __device__ IndexType get(
//    IndexType linearId,
//    const TensorInfo<IndexType>& info) {
//    IndexType offset = 0;

//    // Use static dims
//    for (int i = Dims - 1; i >= 0; --i) {
//      IndexType curDimIndex = linearId % info.sizes[i];
//      IndexType curDimOffset = curDimIndex * info.strides[i];
//      offset += curDimOffset;

//      if (i > 0) {
//        linearId /= info.sizes[i];
//      }
//    }

//    return offset;
//  }
//};

//template <typename IndexType>
//struct IndexToOffset<IndexType, -2> {
//  static __forceinline__ __host__ __device__ IndexType
//    get(IndexType linearId, const TensorInfo<IndexType>& info) {
//    return linearId;
//  }
//};

//template <typename IndexType>
//struct IndexToOffset<IndexType, -1> {
//  static __forceinline__ __host__ __device__ IndexType
//    get(IndexType linearId, const TensorInfo<IndexType>& info) {
//    IndexType offset = 0;

//    // Use dynamic dims
//    for (int i = info.dims - 1; i >= 0; --i) {
//      IndexType curDimIndex = linearId % info.sizes[i];
//      IndexType curDimOffset = curDimIndex * info.strides[i];
//      offset += curDimOffset;

//      linearId /= info.sizes[i];
//    }

//    return offset;
//  }
//};

//template <typename IndexType>
//__device__ __forceinline__ IndexType getLinearBlockId() {
//  return blockIdx.z * gridDim.y * gridDim.x +
//    blockIdx.y * gridDim.x +
//    blockIdx.x;
//}

// Returns true if all linear ID -> offset math can be performed using 32 bit
// unsigned math, which is faster than 64 bit math
bool THCL_canUse32BitIndexMath(THClState* state, THClTensor* t);

// Produces a grid with at least one point per tile
bool THCL_getGridFromTiles(long gridTiles, dim3& grid);

// Determines if the given tensor has overlapping data points (i.e.,
// is there more than one index into the tensor that references the
// same piece of data)?
bool THCL_overlappingIndices(THClState* state, THClTensor* t);

void THCL_checkTensorDims(THClState* state, THClTensor* tensor, int arg);

#endif // THCL_REDUCE_APPLY_UTILS_INC
