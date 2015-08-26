#include <string>

#include "THClDeviceUtils.h"

#define DEFINE_THCLCEILDIV(TYPE) \
TYPE THClCeilDiv(TYPE a, TYPE b) { \
  return (a + b - 1) / b; \
}

DEFINE_THCLCEILDIV(uint32_t);
DEFINE_THCLCEILDIV(uint64_t);
DEFINE_THCLCEILDIV(int32_t);
DEFINE_THCLCEILDIV(int64_t);

//template uint64_t THClCeilDiv<uint64>(uint64_t a, uint64_t b);
//template uint32_t THClCeilDiv<uint32>(uint32_t a, uint32_t b);
//template int64_t THClCeilDiv<int64>(int64_t a, int64_t b);
//template int32_t THClCeilDiv<int32>(int32_t a, int32_t b);

std::string THClDeviceUtils_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClDeviceUtils.cl" )
  // ]]]
  // generated using cog, from THClDeviceUtils.cl:
  const char * kernelSource =  
  "inline {{IndexType}} THClCeilDiv({{IndexType}} a, {{IndexType}} b) {\n" 
  "  return (a + b - 1) / b;\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}


