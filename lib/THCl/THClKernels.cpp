#include "THClKernels.h"
#include "EasyCL.h"
#include "THClTensor.h"
#include <stdexcept>

using namespace std;

THClKernels::THClKernels(THClState *state, CLKernel *kernel) :
  state(state),
  kernel(kernel) {
}
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

