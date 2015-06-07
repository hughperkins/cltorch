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
// #define MAX_CUTORCH_DIMS 25

// Enum that indicates whether tensor arguments are read/write or
// read-only
//enum TensorArgType { ReadWrite, ReadOnly };

// kernel argument that defines tensor layout
typedef struct TensorInfoCl {
  // Extracts size/stride information for the kernel.
  // Successive dimensions can be collapsed if the size/strides match
  // up and thus there are no holes between the dimensions. This is used
  // to reduce the complexity of the problem.
  // The optional `reduceDim` indicates a reduction dimension for the
  // given tensor, so that the output size for this dimension will be 1.

  int sizes[{{MAX_CLNN_DIMS}}];
  int strides[{{MAX_CLNN_DIMS}}];
  int offset;
  int dims;
} TensorInfoCl;
// Contiguous tensors of more than one dimension are collapsed down
// to one tensor
bool TensorInfo_isContiguous( TensorInfoCl tensorInfo ) {
    return (tensorInfo.dims == 1 && tensorInfo.strides[0] == 1);    
}

void op1( global float *out ) {
    {{operation}};
}

// Translate a linear index for the apply to a float* offset;
// specialized on `Dims` to reduce nvcc compilation time
{% for _,dim in ipairs(dims) do %}
int IndexToOffset_{{1000 + dim}}_get( int linearId, TensorInfoCl info) {
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

int IndexToOffset_998_get(int linearId, const TensorInfoCl info) {
    return linearId;
}

int IndexToOffset_999_get(int linearId, const TensorInfoCl info) {
  int offset = 0;

  // Use dynamic dims
  for (int i = info.dims - 1; i >= 0; --i) {
    int curDimIndex = linearId % info.sizes[i];
    int curDimOffset = curDimIndex * info.strides[i];
    offset += curDimOffset;

    linearId /= info.sizes[i];
  }

  return offset;
}

kernel void
THClTensor_pointwiseApply1(global TensorInfoCl *a,
                            global float* a_data,
                             int totalElements) {
  // these are mostly just to help me work out the conversions
  // than to actually use:
  // int blockDim_x = get_local_size(0);
  // int blockIdx_x = get_group_id(0);
  // int threadIdx_x = get_local_id(0);
  // int gridDim_x = get_num_groups(0);
  // blockIdx.x * blockDim.x + threadIdx.x
  //
  // = get_group_id(0) * get_local_size(0) + get_local_id(0)
  //  = get_global_id(0) ?
  //
  // gridDim.x * blockDim.x = get_num_groups(0) * get_local_size(0)
  //                        = get_global_size(0)
  //
  // for (int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  for (int linearIndex = get_global_id(0); // I .... guess?
       linearIndex < totalElements;
       // linearIndex += gridDim.x * blockDim.x) {
       linearIndex += get_global_size(0) /* ? */ ) {
    // Convert `linearIndex` into an offset of `a`
    const int aOffset =
      IndexToOffset_{{1000+adim}}_get(linearIndex, a[0]);

    op1( &(a_data[aOffset + a->offset]) );
  }
}

