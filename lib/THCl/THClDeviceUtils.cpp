#include <string>

#include "THClDeviceUtils.h"

std::string THClDeviceUtils_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClDeviceUtils.cl" )
  // ]]]
  // generated using cog, from THClDeviceUtils.cl:
  const char * kernelSource =  
  "{{IndexType}} THClCeilDiv({{IndexType}} a, {{IndexType}} b) {\n" 
  "  return (a + b - 1) / b;\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}


