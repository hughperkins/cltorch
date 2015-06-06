// OpenCL kernels....

// (Ported from cutorch's THCApply.cuh)

// Maximum number of dimensions allowed for cutorch
#define MAX_CUTORCH_DIMS 25

// Enum that indicates whether tensor arguments are read/write or
// read-only
enum TensorArgType { ReadWrite, ReadOnly };

// kernel argument that defines tensor layout
struct TensorInfo {
  // Extracts size/stride information for the kernel.
  // Successive dimensions can be collapsed if the size/strides match
  // up and thus there are no holes between the dimensions. This is used
  // to reduce the complexity of the problem.
  // The optional `reduceDim` indicates a reduction dimension for the
  // given tensor, so that the output size for this dimension will be 1.

  float* data;
  int sizes[MAX_CUTORCH_DIMS];
  int strides[MAX_CUTORCH_DIMS];
  int dims;
};
// Contiguous tensors of more than one dimension are collapsed down
// to one tensor
bool TensorInfo_isContiguous( TensorInfo *tensorInfo ) {
    return (tensorInfo->dims == 1 && tensorInfo->strides[0] == 1);    
}

float op2( float val1, float val2 ) {
    return {{operation}};
}

// Translate a linear index for the apply to a float* offset;
// specialized on `Dims` to reduce nvcc compilation time
int IndexToOffset_A_get( int linearId, const TensorInfo *info) {
  int offset = 0;

  // Use static dims
  for (int i = {{ADims}} - 1; i >= 0; --i) {
    int curDimIndex = linearId % info.sizes[i];
    int curDimOffset = curDimIndex * info.strides[i];
    offset += curDimOffset;

    if (i > 0) {
      linearId /= info.sizes[i];
    }
  }

  return offset;
}

int IndexToOffset_B_get( int linearId, const TensorInfo *info) {
  int offset = 0;

  // Use static dims
  for (int i = {{BDims}} - 1; i >= 0; --i) {
    int curDimIndex = linearId % info.sizes[i];
    int curDimOffset = curDimIndex * info.strides[i];
    offset += curDimOffset;

    if (i > 0) {
      linearId /= info.sizes[i];
    }
  }

  return offset;
}

int IndexToOffset_C_get( int linearId, const TensorInfo *info) {
  int offset = 0;

  // Use static dims
  for (int i = {{CDims}} - 1; i >= 0; --i) {
    int curDimIndex = linearId % info.sizes[i];
    int curDimOffset = curDimIndex * info.strides[i];
    offset += curDimOffset;

    if (i > 0) {
      linearId /= info.sizes[i];
    }
  }

  return offset;
}


template <typename Op, int ADims, int BDims, int CDims>
kernel void
THClTensor_pointwiseApply3(TensorInfo a,
                             TensorInfo b,
                             TensorInfo c,
                             int totalElements,
                             Op op) {
  for (int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const int aOffset =
      IndexToOffset<int, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const int bOffset =
      IndexToOffset<int, BDims>::get(linearIndex, b);

    // Convert `linearIndex` into an offset of `c`
    const int cOffset =
      IndexToOffset<int, CDims>::get(linearIndex, c);

    a.data[aOffset] = op(b.data[bOffset], c.data[cOffset]);
  }
}


