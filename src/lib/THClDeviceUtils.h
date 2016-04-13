#ifndef THCL_DEVICE_UTILS_INC
#define THCL_DEVICE_UTILS_INC

#include "THClGeneral.h"
#include <string>

/**
   Computes ceil(a / b)
*/
//template <typename T>
//T THClCeilDiv(T a, T b);

#define FLOAT32_MAX_CONSECUTIVE_INT 16777216.0f

#define DECLARE_THCLCEILDIV(TYPE) \
TYPE THClCeilDiv(TYPE a, TYPE b);

DECLARE_THCLCEILDIV(uint32);
DECLARE_THCLCEILDIV(uint64);
DECLARE_THCLCEILDIV(int32);
DECLARE_THCLCEILDIV(int64);

std::string THClDeviceUtils_getKernelTemplate();

#endif // THCL_DEVICE_UTILS_INC

