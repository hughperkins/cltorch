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

{{include_THClReduceApplyUtils}}

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

