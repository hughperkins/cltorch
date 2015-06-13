#include "THClTensorInfoCl.h"

#include <iostream>
using namespace std;

std::string THClTensorInfoCl_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClTensorInfoCl.cl" )
  // ]]]
  // generated using cog, from THClTensorInfoCl.cl:
  const char * kernelSource =  
  "// kernel argument that defines tensor layout\n" 
  "typedef struct TensorInfoCl {\n" 
  "  // Extracts size/stride information for the kernel.\n" 
  "  // Successive dimensions can be collapsed if the size/strides match\n" 
  "  // up and thus there are no holes between the dimensions. This is used\n" 
  "  // to reduce the complexity of the problem.\n" 
  "  // The optional `reduceDim` indicates a reduction dimension for the\n" 
  "  // given tensor, so that the output size for this dimension will be 1.\n" 
  "\n" 
  "  {{IndexType}} sizes[{{MAX_CLTORCH_DIMS}}];\n" 
  "  {{IndexType}} strides[{{MAX_CLTORCH_DIMS}}];\n" 
  "  {{IndexType}} offset;\n" 
  "  int dims;\n" 
  "} TensorInfoCl;\n" 
  "// Contiguous tensors of more than one dimension are collapsed down\n" 
  "// to one tensor\n" 
  "bool TensorInfo_isContiguous( TensorInfoCl tensorInfo ) {\n" 
  "    return (tensorInfo.dims == 1 && tensorInfo.strides[0] == 1);\n" 
  "}\n" 
  "\n" 
  "// Translate a linear index for the apply to a float* offset;\n" 
  "// specialized on `Dims` to reduce nvcc compilation time\n" 
  "{% for _,dim in ipairs(dims) do %}\n" 
  "{{IndexType}} IndexToOffset_{{1000 + dim}}_get( {{IndexType}} linearId, TensorInfoCl info) {\n" 
  "  {{IndexType}} offset = 0;\n" 
  "\n" 
  "  // Use static dims\n" 
  "  for (int i = {{dim}} - 1; i >= 0; --i) {\n" 
  "    {{IndexType}} curDimIndex = linearId % info.sizes[i];\n" 
  "    {{IndexType}} curDimOffset = curDimIndex * info.strides[i];\n" 
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
  "{{IndexType}} IndexToOffset_998_get({{IndexType}} linearId, const TensorInfoCl info) {\n" 
  "    return linearId;\n" 
  "}\n" 
  "\n" 
  "{{IndexType}} IndexToOffset_999_get({{IndexType}} linearId, const TensorInfoCl info) {\n" 
  "  {{IndexType}} offset = 0;\n" 
  "\n" 
  "  // Use dynamic dims\n" 
  "  for (int i = info.dims - 1; i >= 0; --i) {\n" 
  "    {{IndexType}} curDimIndex = linearId % info.sizes[i];\n" 
  "    {{IndexType}} curDimOffset = curDimIndex * info.strides[i];\n" 
  "    offset += curDimOffset;\n" 
  "\n" 
  "    linearId /= info.sizes[i];\n" 
  "  }\n" 
  "\n" 
  "  return offset;\n" 
  "}\n" 
  "\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}



