#include "THClApply.h"

// Implementation of copyIgnoringOverlaps, defined after pointwiseApply2.
void THClTensor_copyIgnoringOverlaps(THClState* state,
                                       THClTensor* dst,
                                       THClTensor* src) {
  THClTensor_pointwiseApply2(state, dst, src, CopyOp(),
                               ReadOnly, // ignore overwrites
                               ReadOnly);
}

std::string getApplyDv2_template() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClApply.cl" )
  // ]]]
  // generated using cog, from THClApply.cl:
  const char * kernelSource =  
  "// OpenCL kernels....\n" 
  "\n" 
  "// expected templated values:\n" 
  "// dims (vector of unique dimension values)\n" 
  "// operation\n" 
  "// dim1\n" 
  "// dim2\n" 
  "// dim3\n" 
  "// ... dimD\n" 
  "// num_input_tensors\n" 
  "// include_scalar_input\n" 
  "//\n" 
  "// maybe should add:\n" 
  "// IndexType (hardcoded to int for now)\n" 
  "// MAX_CUTORCH_DIMS (hardcoded to 25 for now)\n" 
  "\n" 
  "// (Ported from cutorch's THCApply.cuh)\n" 
  "\n" 
  "// Maximum number of dimensions allowed for cutorch\n" 
  "// #define MAX_CUTORCH_DIMS 25\n" 
  "\n" 
  "// Enum that indicates whether tensor arguments are read/write or\n" 
  "// read-only\n" 
  "//enum TensorArgType { ReadWrite, ReadOnly };\n" 
  "\n" 
  "// not used by this kernel, but used by THClReduceApplyUtils...\n" 
  "static float reduceOp(float _in1, float _in2) {\n" 
  "  return 0;\n" 
  "}\n" 
  "\n" 
  "{{include_THClReduceApplyUtils}}\n" 
  "\n" 
  "{%\n" 
  " total_opsize = num_tensors\n" 
  " if include_scalar_input then\n" 
  "      total_opsize = total_opsize + 1\n" 
  "   end\n" 
  " %}\n" 
  "\n" 
  "static void op( global float *out\n" 
  "  {% for i=1,(num_tensors-1) do %}\n" 
  "  , global float *in{{i}}\n" 
  "  {% end %}\n" 
  "  {% for i=1,(num_scalars) do %}\n" 
  "  , float val{{i}}\n" 
  "  {% end %}\n" 
  ") {\n" 
  "    {{operation}};\n" 
  "}\n" 
  "\n" 
  "kernel void\n" 
  "THClTensor_pointwiseApplyD(\n" 
  "   {% for input_idx=1,num_tensors do %}\n" 
  "    global TensorInfoCl *info_{{input_idx}},\n" 
  "    global float*data_{{input_idx}},\n" 
  "   {% end %}\n" 
  "   {% for i=1,num_scalars do %}\n" 
  "   float val{{i}},\n" 
  "   {% end %}\n" 
  "   int totalElements) {\n" 
  "  for (int linearIndex = get_global_id(0);\n" 
  "       linearIndex < totalElements;\n" 
  "       linearIndex += get_global_size(0)) {\n" 
  "    {% for input_idx=1,num_tensors do %}\n" 
  "    // Convert `linearIndex` into an offset of `a`\n" 
  "    const int offset{{input_idx}} =\n" 
  "      IndexToOffset_{{1000+loadstring('return dim' .. input_idx)()}}_get(linearIndex, info_{{input_idx}}[0]);\n" 
  "    {% end %}\n" 
  "\n" 
  "    op(\n" 
  "      {% for input_idx=1,num_tensors do %}\n" 
  "         {% if input_idx > 1 then %} , {% end %}\n" 
  "         &(data_{{input_idx}}[offset{{input_idx}}])\n" 
  "      {% end %}\n" 
  "      {% for i=1,num_scalars do %}\n" 
  "      , val{{i}}\n" 
  "      {% end %}\n" 
  "    );\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

