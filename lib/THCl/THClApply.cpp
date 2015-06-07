#include "THClApply.h"

// Implementation of copyIgnoringOverlaps, defined after pointwiseApply2.
void THClTensor_copyIgnoringOverlaps(THClState* state,
                                       THClTensor* dst,
                                       THClTensor* src) {
  THClTensor_pointwiseApply2(state, dst, src, CopyOp<float>(),
                               ReadOnly, // ignore overwrites
                               ReadOnly);
}

std::string getApply2_template() {
    // [[[cog
    // import stringify
    // stringify.write_kernel( "kernel", "THClApply2.cl" )
    // ]]]
    // generated using cog, from THClApply2.cl:
    const char * kernelSource =  
    "// OpenCL kernels....\n" 
    "\n" 
    "// expected templated values:\n" 
    "// dims (vector of unique dimension values)\n" 
    "// operation\n" 
    "// adim\n" 
    "// bdim\n" 
    "// cdim\n" 
    "//\n" 
    "// maybe should add:\n" 
    "// IndexType (hardcoded to int for now)\n" 
    "// MAX_CUTORCH_DIMS (hardcoded to 25 for now)\n" 
    "\n" 
    "// (Ported from cutorch's THCApply.cuh)\n" 
    "\n" 
    "// Maximum number of dimensions allowed for cutorch\n" 
    "#define MAX_CUTORCH_DIMS 25\n" 
    "\n" 
    "// Enum that indicates whether tensor arguments are read/write or\n" 
    "// read-only\n" 
    "//enum TensorArgType { ReadWrite, ReadOnly };\n" 
    "\n" 
    "// kernel argument that defines tensor layout\n" 
    "struct TensorInfo {\n" 
    "  // Extracts size/stride information for the kernel.\n" 
    "  // Successive dimensions can be collapsed if the size/strides match\n" 
    "  // up and thus there are no holes between the dimensions. This is used\n" 
    "  // to reduce the complexity of the problem.\n" 
    "  // The optional `reduceDim` indicates a reduction dimension for the\n" 
    "  // given tensor, so that the output size for this dimension will be 1.\n" 
    "\n" 
    "  float* data;\n" 
    "  int sizes[{{MAX_CUTORCH_DIMS}}];\n" 
    "  int strides[{{MAX_CUTORCH_DIMS}}];\n" 
    "  int dims;\n" 
    "};\n" 
    "// Contiguous tensors of more than one dimension are collapsed down\n" 
    "// to one tensor\n" 
    "bool TensorInfo_isContiguous( TensorInfo tensorInfo ) {\n" 
    "    return (tensorInfo.dims == 1 && tensorInfo.strides[0] == 1);\n" 
    "}\n" 
    "\n" 
    "void op2( float *out, float *val1 ) {\n" 
    "    return {{operation}};\n" 
    "}\n" 
    "\n" 
    "// Translate a linear index for the apply to a float* offset;\n" 
    "// specialized on `Dims` to reduce nvcc compilation time\n" 
    "{% for _,dim in ipairs(dims) do %}\n" 
    "int IndexToOffset_{{1000 + dim}}_get( int linearId, const TensorInfo info) {\n" 
    "  int offset = 0;\n" 
    "\n" 
    "  // Use static dims\n" 
    "  for (int i = {{dim}} - 1; i >= 0; --i) {\n" 
    "    int curDimIndex = linearId % info.sizes[i];\n" 
    "    int curDimOffset = curDimIndex * info.strides[i];\n" 
    "    offset += curDimOffset;\n" 
    "\n" 
    "    if (i > 0) {\n" 
    "      linearId /= info.sizes[i];\n" 
    "    }\n" 
    "  }\n" 
    "\n" 
    "  return offset;\n" 
    "}\n" 
    "{% end %}\n" 
    "\n" 
    "kernel void\n" 
    "THClTensor_pointwiseApply2(TensorInfo a,\n" 
    "                             TensorInfo b,\n" 
    "                             int totalElements {\n" 
    "  for (int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;\n" 
    "       linearIndex < totalElements;\n" 
    "       linearIndex += gridDim.x * blockDim.x) {\n" 
    "    // Convert `linearIndex` into an offset of `a`\n" 
    "    const int aOffset =\n" 
    "      IndexToOffset_{{1000+adim}}_get(linearIndex, a);\n" 
    "\n" 
    "    // Convert `linearIndex` into an offset of `b`\n" 
    "    const int bOffset =\n" 
    "      IndexToOffset_{{1000+bdim}}_get(linearIndex, b);\n" 
    "\n" 
    "    op2( &(a.data[aOffset]), &(b.data[bOffset]));\n" 
    "  }\n" 
    "}\n" 
    "\n" 
    "\n" 
    "";
    // [[[end]]]
}

