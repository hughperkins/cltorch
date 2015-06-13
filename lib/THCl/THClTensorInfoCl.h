#pragma once

#include <string>
#include <stdexcept>

#include "THClReduceApplyUtils.h"
#include "util/easycl_stringhelper.h"

std::string THClTensorInfoCl_getKernelTemplate();

typedef struct TensorInfoCl {
  TensorInfoCl( TensorInfo<unsigned int> info ) {
    dims = info.dims;
    if( info.offset > ( 1l << 30 ) ) {
      throw std::runtime_error("size " + easycl::toString(info.offset) + " out of bounds");
    }
    offset = (int)info.offset;
    for( int i = 0; i < dims; i++ ) {
      sizes[i] = info.sizes[i];
      strides[i] = info.strides[i];
    }
  }
  TensorInfoCl( TensorInfo<unsigned long> info ) {
    dims = info.dims;
    if( info.offset > ( 1l << 30 ) ) {
      throw std::runtime_error("size " + easycl::toString(info.offset) + " out of bounds");
    }
    offset = (int)info.offset;
    for( int i = 0; i < dims; i++ ) {
      if( info.sizes[i] > ( 1l << 31 ) ) {
        throw std::runtime_error("size " + easycl::toString(info.sizes[i]) + " out of bounds");
      }
      sizes[i] = info.sizes[i];
      strides[i] = info.strides[i];
    }
  }
  unsigned int sizes[MAX_CLTORCH_DIMS];
  unsigned int strides[MAX_CLTORCH_DIMS];
  int offset;
  int dims;
} TensorInfoCl;


