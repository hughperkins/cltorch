// OpenCL kernels....

// expected templated values:
// dims (vector of unique dimension values)
// operation
// dim1
// dim2
// dim3
// ... dimD
// num_input_tensors
// include_scalar_input
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

{%
 total_opsize = num_tensors
 if include_scalar_input then 
      total_opsize = total_opsize + 1
   end
 %}

void op( global float *out
  {% for i=1,(num_tensors-1) do %}
  , global float *in{{i}}
  {% end %}
  {% for i=1,(num_scalars) do %}
  , float val{{i}}
  {% end %}
) {
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
THClTensor_pointwiseApplyD(
   {% for input_idx=1,num_tensors do %}
    global TensorInfoCl *info_{{input_idx}},
    global float*data_{{input_idx}},
   {% end %}
   {% for i=1,num_scalars do %}
   float val{{i}},
   {% end %}
   int totalElements) {
  for (int linearIndex = get_global_id(0);
       linearIndex < totalElements;
       linearIndex += get_global_size(0) /* ? */ ) {
    {% for input_idx=1,num_tensors do %}
    // Convert `linearIndex` into an offset of `a`
    const int offset{{input_idx}} =
      IndexToOffset_{{1000+loadstring('return dim' .. input_idx)()}}_get(linearIndex, info_{{input_idx}}[0]);
    {% end %}

    op( 
      {% for input_idx=1,num_tensors do %}
         {% if input_idx > 1 then %} , {% end %}
         &(data_{{input_idx}}[offset{{input_idx}} + info_{{input_idx}}->offset])
      {% end %}
      {% for i=1,num_scalars do %}
      , val{{i}}
      {% end %}
    );
  }
}

