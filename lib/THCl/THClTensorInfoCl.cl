// kernel argument that defines tensor layout
typedef struct TensorInfoCl {
  // Extracts size/stride information for the kernel.
  // Successive dimensions can be collapsed if the size/strides match
  // up and thus there are no holes between the dimensions. This is used
  // to reduce the complexity of the problem.
  // The optional `reduceDim` indicates a reduction dimension for the
  // given tensor, so that the output size for this dimension will be 1.

  int sizes[{{MAX_CLTORCH_DIMS}}];
  int strides[{{MAX_CLTORCH_DIMS}}];
  int offset;
  int dims;
} TensorInfoCl;
// Contiguous tensors of more than one dimension are collapsed down
// to one tensor
bool TensorInfo_isContiguous( TensorInfoCl tensorInfo ) {
    return (tensorInfo.dims == 1 && tensorInfo.strides[0] == 1);    
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


