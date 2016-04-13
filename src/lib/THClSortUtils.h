//// from lib/THC/THCSortUtils.cuh:

#ifndef THCL_SORT_UTILS_INC
#define THCL_SORT_UTILS_INC

//#include "THClReduceApplyUtils.cuh"

//// Collection of kernel sort routines
//template <typename T>
//struct LTComp {
//  __device__ inline bool operator()(const T& a, const T& b) const {
//    return (a < b);
//  }
//};

//template <typename T>
//struct GTComp {
//  __device__ inline bool operator()(const T& a, const T& b) const {
//    return (a > b);
//  }
//};

std::string THClSortUtils_getKernelTemplate();

#endif // THCL_SORT_UTILS_INC

