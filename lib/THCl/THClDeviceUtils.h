#ifndef THCL_DEVICE_UTILS_INC
#define THCL_DEVICE_UTILS_INC

#include <string>

/**
   Computes ceil(a / b)
*/
template <typename T>
inline T THClCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

std::string THClDeviceUtils_getKernelTemplate();

#endif // THCL_DEVICE_UTILS_INC

