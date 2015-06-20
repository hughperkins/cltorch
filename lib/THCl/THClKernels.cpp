#include "THClKernels.h"
#include "EasyCL.h"
#include "THClTensor.h"

THClKernels::THClKernels(THClState *state, CLKernel *kernel) :
  state(state),
  kernel(kernel) {
}
THClKernels *THClKernels::in(THClTensor *tensor) {
  kernel->in(THClTensor_wrapper(state, tensor));
  kernel->in((int)THClTensor_storageOffset(state, tensor));
  return this;
}
THClKernels *THClKernels::inout(THClTensor *tensor) {
  kernel->inout(THClTensor_wrapper(state, tensor));
  kernel->in((int)THClTensor_storageOffset(state, tensor));
  return this;
}
THClKernels *THClKernels::out(THClTensor *tensor) {
  kernel->out(THClTensor_wrapper(state, tensor));
  kernel->in((int)THClTensor_storageOffset(state, tensor));
  return this;
}
THClKernels *THClKernels::in(int value) {
  kernel->in(value);
  return this;
}
THClKernels *THClKernels::in(float value) {
  kernel->in(value);
  return this;
}
THClKernels *THClKernels::in(CLWrapper *wrapper) {
  kernel->in(wrapper);
  return this;
}
THClKernels *THClKernels::inout(CLWrapper *wrapper) {
  kernel->inout(wrapper);
  return this;
}
THClKernels *THClKernels::out(CLWrapper *wrapper) {
  kernel->out(wrapper);
  return this;
}

