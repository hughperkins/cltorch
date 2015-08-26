#ifndef THCL_DEVICE_UTILS_INC
#define THCL_DEVICE_UTILS_INC

#include "THClGeneral.h"
#include <string>

/**
   Computes ceil(a / b)
*/
//template <typename T>
//T THClCeilDiv(T a, T b);

#define DECLARE_THCLCEILDIV(TYPE) \
TYPE THClCeilDiv(TYPE a, TYPE b);

DECLARE_THCLCEILDIV(uint32_t);
DECLARE_THCLCEILDIV(uint64_t);
DECLARE_THCLCEILDIV(int32_t);
DECLARE_THCLCEILDIV(int64_t);

std::string THClDeviceUtils_getKernelTemplate();

#endif // THCL_DEVICE_UTILS_INC

