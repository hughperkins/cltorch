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

{%
 local total_opsize = num_tensors
 if include_scalar_input then 
      total_opsize = total_opsize + 1
   end
 %}

inline void op( global float *out
  {% for t=1,(num_tensors-1) do %}
  , global float *in{{t}}
  {% end %}
  {% for s=1,(num_scalars) do %}
  , float val{{s}}
  {% end %}
   {% for pt=1,num_point_tensors do %}
   , global float *pointTensor{{pt}}
   {% end %}
) {
    {{operation}};
}

kernel void
THClTensor_pointwiseApplyD(
   {% for t=1,num_tensors do %}
    int offset_{{t}},
    {% local thisdims = loadstring('return dims' .. t)() %}
    {% for d=1,thisdims do %}
      int size_{{t}}_{{d}},
      int stride_{{t}}_{{d}},
    {% end %}
    global float*data_{{t}},
   {% end %}
   {% for i=1,num_scalars do %}
   float val{{i}},
   {% end %}
   {% for i=1,num_point_tensors do %}
   global float *pointTensor{{i}},
   {% end %}
   int totalElements) {
   int linearIndex = get_global_id(0);
   if(linearIndex < totalElements ) {
    {% if thisdims ~= -2 then %}
    int thisLinearId;
    {% end %}
    {% for t=1,num_tensors do %}
      {% local thisdims = loadstring('return dims' .. t)() %}
      {% if thisdims == -2 then %}
         int derived_offset_{{t}} = linearIndex + offset_{{t}};
      {% else %}
         {{IndexType}} derived_offset_{{t}} = offset_{{t}};
         thisLinearId = linearIndex;
        {% for d=thisdims,1,-1 do %}  // bake this in....
          derived_offset_{{t}} += (thisLinearId % size_{{t}}_{{d}}) * stride_{{t}}_{{d}};
          {% if d > 0 then %}
            thisLinearId /= size_{{t}}_{{d}};
          {% end %}
        {% end %}

      {% end %}
    {% end %}

    op( 
      {% for t=1,num_tensors do %}
         {% if t > 1 then %} , {% end %}
         &(data_{{t}}[derived_offset_{{t}}])
      {% end %}

      {% for s=1,num_scalars do %}
      , val{{s}}
      {% end %}

       {% for pt=1,num_point_tensors do %}
       , pointTensor{{pt}}
       {% end %}
    );
  }
}

