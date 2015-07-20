#include "THClKernels.h"
#include "EasyCL.h"
#include "THClTensor.h"
#include <stdexcept>
#include "THClReduceApplyUtils.h"
#include "CLKernel_structs.h"

#include <iostream>
using namespace std;

// Constructor
THClKernels::THClKernels(THClState *state, CLKernel *kernel) :
  state(state),
  kernel(kernel) {
}
THClKernels::~THClKernels() {
  for( int i = 0; i < (int)tensorInfoCls.size(); i++ ) {
    delete tensorInfoCls[i];
  }
}
// CLTensors =====================
THClKernels *THClKernels::in(THClTensor *tensor) {
  try {
    kernel->in(THClTensor_wrapper(state, tensor));
    kernel->in((int)THClTensor_storageOffset(state, tensor));
  } catch( runtime_error &e ) {
    THError(e.what());
  }
  return this;
}
THClKernels *THClKernels::inout(THClTensor *tensor) {
  try {
    kernel->inout(THClTensor_wrapper(state, tensor));
    kernel->in((int)THClTensor_storageOffset(state, tensor));
  } catch( runtime_error &e ) {
    THError(e.what());
  }
  return this;
}
THClKernels *THClKernels::out(THClTensor *tensor) {
  try {
    kernel->out(THClTensor_wrapper(state, tensor));
    kernel->in((int)THClTensor_storageOffset(state, tensor));
  } catch( runtime_error &e ) {
    THError(e.what());
  }
  return this;
}
// CLTensors v2 =====================
THClKernels *THClKernels::inv2(THClTensor *tensor) {
  try {
    TensorInfoCl *tensorInfoCl = new TensorInfoCl();
    initTensorInfoCl(tensorInfoCl, tensor);
    kernel->in(1, tensorInfoCl);
    kernel->in(THClTensor_wrapper(state, tensor));
    tensorInfoCls.push_back(tensorInfoCl);
  } catch( runtime_error &e ) {
    THError(e.what());
  }
  return this;
}
THClKernels *THClKernels::inoutv2(THClTensor *tensor) {
  try {
    TensorInfoCl *tensorInfoCl = new TensorInfoCl();
    initTensorInfoCl(tensorInfoCl, tensor);
    kernel->in(1, tensorInfoCl);
    kernel->inout(THClTensor_wrapper(state, tensor));
    tensorInfoCls.push_back(tensorInfoCl);
  } catch( runtime_error &e ) {
    THError(e.what());
  }
  return this;
}
THClKernels *THClKernels::outv2(THClTensor *tensor) {
  try {
    TensorInfoCl *tensorInfoCl = new TensorInfoCl();
    initTensorInfoCl(tensorInfoCl, tensor);
    kernel->in(1, tensorInfoCl);
    kernel->out(THClTensor_wrapper(state, tensor));
    tensorInfoCls.push_back(tensorInfoCl);
  } catch( runtime_error &e ) {
    THError(e.what());
  }
  return this;
}
// scalars ==================
THClKernels *THClKernels::in(int value) {
  try {
    kernel->in(value);
  } catch( runtime_error &e ) {
    THError(e.what());
  }
  return this;
}
THClKernels *THClKernels::in(float value) {
  try {
    kernel->in(value);
  } catch( runtime_error &e ) {
    THError(e.what());
  }
  return this;
}
// CLTensorInfos ================
template< typename IndexType >
THClKernels *THClKernels::in(TensorInfo<IndexType>tensorInfo) {
  TensorInfoCl *tensorInfoCl = new TensorInfoCl();
  initTensorInfoCl(tensorInfoCl, tensorInfo);
  kernel->in(1, tensorInfoCl);
  kernel->in(tensorInfo.wrapper);
  tensorInfoCls.push_back(tensorInfoCl);
  return this;
}
template< typename IndexType >
THClKernels *THClKernels::inout(TensorInfo<IndexType>tensorInfo) {
  TensorInfoCl *tensorInfoCl = new TensorInfoCl();
  initTensorInfoCl(tensorInfoCl, tensorInfo);
  kernel->in(1, tensorInfoCl);
  kernel->inout(tensorInfo.wrapper);
  tensorInfoCls.push_back(tensorInfoCl);
  return this;
}
template< typename IndexType >
THClKernels *THClKernels::out(TensorInfo<IndexType>tensorInfo) {
  TensorInfoCl *tensorInfoCl = new TensorInfoCl();
  initTensorInfoCl(tensorInfoCl, tensorInfo);
  if( !tensorInfo.wrapper->isOnDevice() ) {
    tensorInfo.wrapper->createOnDevice();
  }
  kernel->in(1, tensorInfoCl);
  kernel->out(tensorInfo.wrapper);
  tensorInfoCls.push_back(tensorInfoCl);
  return this;
}
// CLWrapper ===============
THClKernels *THClKernels::in(CLWrapper *wrapper) {
  try {
    kernel->in(wrapper);
  } catch( runtime_error &e ) {
    THError(e.what());
  }
  return this;
}
THClKernels *THClKernels::inout(CLWrapper *wrapper) {
  try {
    kernel->inout(wrapper);
  } catch( runtime_error &e ) {
    THError(e.what());
  }
  return this;
}
THClKernels *THClKernels::out(CLWrapper *wrapper) {
  try {
    if( !wrapper->isOnDevice() ) {
      wrapper->createOnDevice();
    }
    kernel->out(wrapper);
  } catch( runtime_error &e ) {
    THError(e.what());
  }
  return this;
}
void THClKernels::run(dim3 grid, dim3 block) {
  dim3 global_ws;
  for( int i = 0; i < 3; i++ ) {
      global_ws.vec[i] = grid.vec[i] * block.vec[i];
  }
  try {
    kernel->run(3, global_ws.as_size_t(), block.as_size_t());
  } catch( runtime_error &e ) {
    cout << e.what() << endl;
    THError(e.what());
  }
}
// locals ==================
THClKernels *THClKernels::localFloats(int count) {
  try {
    kernel->localFloats(count);
  } catch( runtime_error &e ) {
    THError(e.what());
  }
  return this;
}

// template instantiations ====================
#define DECLARE_THCLKERNELS(IndexType) \
template \
THClKernels *THClKernels::in<IndexType>(TensorInfo<IndexType>tensorInfo); \
template \
THClKernels *THClKernels::inout<IndexType>(TensorInfo<IndexType>tensorInfo); \
template \
THClKernels *THClKernels::out<IndexType>(TensorInfo<IndexType>tensorInfo);

DECLARE_THCLKERNELS(uint32);
DECLARE_THCLKERNELS(uint64);

template CLKernel *CLKernel::in<>(int N, const TensorInfoCl *data);
template CLKernel *CLKernel::inout<>(int N, const TensorInfoCl *data);
template CLKernel *CLKernel::out<>(int N, const TensorInfoCl *data);

