#include <string>

#include "THClDeviceUtils.h"

#define DEFINE_THCLCEILDIV(TYPE) \
TYPE THClCeilDiv(TYPE a, TYPE b) { \
  return (a + b - 1) / b; \
}

DEFINE_THCLCEILDIV(uint32);
DEFINE_THCLCEILDIV(uint64);
DEFINE_THCLCEILDIV(int32);
DEFINE_THCLCEILDIV(int64);

//template uint64 THClCeilDiv<uint64>(uint64 a, uint64 b);
//template uint32 THClCeilDiv<uint32>(uint32 a, uint32 b);
//template int64 THClCeilDiv<int64>(int64 a, int64 b);
//template int32 THClCeilDiv<int32>(int32 a, int32 b);

std::string THClDeviceUtils_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClDeviceUtils.cl" )
  // ]]]
  // generated using cog, from THClDeviceUtils.cl:
  const char * kernelSource =  
  "static {{IndexType}} THClCeilDiv({{IndexType}} a, {{IndexType}} b) {\n" 
  "  return (a + b - 1) / b;\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}


