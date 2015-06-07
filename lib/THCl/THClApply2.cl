// OpenCL kernels....

// expected templated values:
// dims (vector of unique dimension values)
// operation
// adim
// bdim
// cdim
//
// maybe should add:
// IndexType (hardcoded to int for now)
// MAX_CUTORCH_DIMS (hardcoded to 25 for now)

// (Ported from cutorch's THCApply.cuh)

// Maximum number of dimensions allowed for cutorch
#define MAX_CUTORCH_DIMS 25

// Enum that indicates whether tensor arguments are read/write or
// read-only
//enum TensorArgType { ReadWrite, ReadOnly };

// kernel argument that defines tensor layout
struct TensorInfo {
  // Extracts size/stride information for the kernel.
  // Successive dimensions can be collapsed if the size/strides match
  // up and thus there are no holes between the dimensions. This is used
  // to reduce the complexity of the problem.
  // The optional `reduceDim` indicates a reduction dimension for the
  // given tensor, so that the output size for this dimension will be 1.

  int sizes[{{MAX_CLNN_DIMS}}];
  int strides[{{MAX_CLNN_DIMS}}];
  int dims;
};
// Contiguous tensors of more than one dimension are collapsed down
// to one tensor
bool TensorInfo_isContiguous( struct TensorInfo tensorInfo ) {
    return (tensorInfo.dims == 1 && tensorInfo.strides[0] == 1);    
}

void op2( float *out, float *in1 ) {
    {{operation}};
}

// Translate a linear index for the apply to a float* offset;
// specialized on `Dims` to reduce nvcc compilation time
{% for _,dim in ipairs(dims) do %}
int IndexToOffset_{{1000 + dim}}_get( int linearId, const struct TensorInfo info) {
  int offset = 0;

  // Use static dims
  for (int i = {{dim}} - 1; i >= 0; --i) {
    int curDimIndex = linearId % info.sizes[i];
    int curDimOffset = curDimIndex * info.strides[i];
    offset += curDimOffset;

    if (i > 0) {
      linearId /= info.sizes[i];
    }
  }

  return offset;
}
{% end %}

kernel void
THClTensor_pointwiseApply2(struct TensorInfo a,
                             struct TensorInfo b,
                            global float* a_data,
                            global float*b_data,
                             int totalElements) {
  for (int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const int aOffset =
      IndexToOffset_{{1000+adim}}_get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const int bOffset =
      IndexToOffset_{{1000+bdim}}_get(linearIndex, b);

    op2( &(a_data[aOffset]), &(b_data[bOffset]));
  }
}


