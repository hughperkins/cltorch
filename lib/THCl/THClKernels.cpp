#include "THClKernels.h"
#include "EasyCL.h"
#include "THClTensor.h"
#include <stdexcept>
#include "THClReduceApplyUtils.h"
#include "CLKernel_structs.h"

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
  kernel->in(THClTensor_wrapper(state, tensor));
  try {
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
  TensorInfoCl *tensorInfoCl = new TensorInfoCl(tensorInfo);
  kernel->in(1, tensorInfoCl);
  kernel->in(tensorInfo.wrapper);
  tensorInfoCls.push_back(tensorInfoCl);
  return this;
}
template< typename IndexType >
THClKernels *THClKernels::inout(TensorInfo<IndexType>tensorInfo) {
  TensorInfoCl *tensorInfoCl = new TensorInfoCl(tensorInfo);
  kernel->in(1, tensorInfoCl);
  kernel->inout(tensorInfo.wrapper);
  tensorInfoCls.push_back(tensorInfoCl);
  return this;
}
template< typename IndexType >
THClKernels *THClKernels::out(TensorInfo<IndexType>tensorInfo) {
  TensorInfoCl *tensorInfoCl = new TensorInfoCl(tensorInfo);
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
  kernel->in(wrapper);
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
  kernel->run(3, global_ws.as_size_t(), block.as_size_t());
}
// locals ==================
THClKernels *THClKernels::localFloats(int count) {
  kernel->localFloats(count);
}

// template instantiations ====================
#define DECLARE_THCLKERNELS(IndexType) \
template \
THClKernels *THClKernels::in<IndexType>(TensorInfo<IndexType>tensorInfo); \
template \
THClKernels *THClKernels::inout<IndexType>(TensorInfo<IndexType>tensorInfo); \
template \
THClKernels *THClKernels::out<IndexType>(TensorInfo<IndexType>tensorInfo);

DECLARE_THCLKERNELS(unsigned int);
DECLARE_THCLKERNELS(unsigned long);

template CLKernel *CLKernel::in<>(int N, const TensorInfoCl *data);
template CLKernel *CLKernel::inout<>(int N, const TensorInfoCl *data);
template CLKernel *CLKernel::out<>(int N, const TensorInfoCl *data);

