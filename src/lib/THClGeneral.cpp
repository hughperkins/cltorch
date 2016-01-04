#include "THClGeneral.h"
#include "TH.h"

#include <stdio.h>
#include "EasyCL.h"
#include <clBLAS.h>
#include "DeviceInfo.h"

//using namespace easycl;

//#include "THCTensorRandom.h"
//#include "THCBlas.h"
//#include "THCAllocator.h"

/* Size of scratch space available in global memory per each SM + stream */
#define FLOATS_PER_SCRATCH_SPACE 4
#define GLOBAL_SCRATCH_SPACE_PER_SM_STREAM (FLOATS_PER_SCRATCH_SPACE) * sizeof(float)

static void initializeState(THClState *state) {
  if(state->initialized) {
    return;
  }
  state->initialized = 1;
  if(state->allowNonGpus) {
    state->allocatedDevices = easycl::DevicesInfo::getNumDevices();
  } else {
    state->allocatedDevices = easycl::DevicesInfo::getNumGpus();
  }
  state->clByDevice = new EasyCL *[state->allocatedDevices];
  state->scratchSpaceByDevice = new THClScratchSpace *[state->allocatedDevices];
  state->trace = 0;
  state->detailedTimings = 0;
  state->addFinish = 0;
//  state->workgroupSizeByDevice = new int[state->allocatedDevices];
  state->deviceInfoByDevice = (DeviceInfo **)new easycl::DeviceInfo *[state->allocatedDevices];
  for(int i = 0; i < state->allocatedDevices; i++) {
    state->clByDevice[i] = 0;
    state->scratchSpaceByDevice[i] = 0;
    state->deviceInfoByDevice[i] = 0;
  }
  state->currentDevice = 0;
  //state->cl = EasyCL::createForFirstGpuOtherwiseCpu(); // obviously this should change...

  cl_int err;

  err = clblasSetup();
  if (err != CL_SUCCESS) {
    THError("clblasSetup() failed with %d", err);
  }
}
void THClSetAllowNonGpus(THClState *state, int allowNonGpus) {
  if(state->initialized) {
    THError("cannot set allowNonGpus after initialization done");
  } else {
    state->allowNonGpus = allowNonGpus;
  }
}
void THClInit(THClState* state)
{
    state->initialized = 0;
    state->allowNonGpus = 0;
    state->trace = 0;
    state->detailedTimings = 0;
    state->addFinish = 0;
    state->currentDevice = 0;
    state->allocatedDevices = 0;

    state->clByDevice = 0;
    state->scratchSpaceByDevice = 0;
    state->deviceInfoByDevice = 0;
}

void THClShutdown(THClState* state)
{
  if(state->initialized == 0) {
      return;
  }

  clblasTeardown();
  for( int i = 0; i < state->allocatedDevices; i++ ) {
    delete state->clByDevice[i];
    delete state->scratchSpaceByDevice[i]->wrapper;
    delete[] state->scratchSpaceByDevice[i]->data;
    delete (easycl::DeviceInfo*)state->deviceInfoByDevice[i];
  }
  delete[] (easycl::DeviceInfo**)state->deviceInfoByDevice;
  delete[] state->clByDevice;
  delete[] state->scratchSpaceByDevice;
  state->initialized = 0;
//  delete[] state->workgroupSizeByDevice

  printf("THClShutdown() done\n");
  printf("*******************************************\n");
}

std::ostream &operator<<( std::ostream &os, const dim3 &obj ) {
  os << "dim3{" << obj.vec[0] << ", " << obj.vec[1] << ", " << obj.vec[2] << "}";
  return os;
}

int THClState_getNumDevices(THClState* state) {
  if(state->initialized == 0) {
    initializeState(state);
  }
  return state->allocatedDevices;
}
void THClState_setDevice(THClState* state, int device) {
  if(state->initialized == 0) {
    initializeState(state);
  }
  state->currentDevice = device;
}
int THClState_getDevice(THClState* state) {
  if(state->initialized == 0) {
    initializeState(state);
  }
  return state->currentDevice;
}
EasyCL *THClState_getCl(THClState* state ) {
  if(state->initialized == 0) {
    initializeState(state);
  }
  return THClState_getClv2(state, state->currentDevice);
}
EasyCL *THClState_getCl(THClState* state, int *p_device) {
  if(state->initialized == 0) {
    initializeState(state);
  }
  if( p_device != 0 ) {
    *p_device = state->currentDevice;
  }
  return THClState_getClv2(state, state->currentDevice);
}
EasyCL *THClState_getClv2(THClState* state, int device) {
  if(!state->initialized) {
    initializeState(state);
  }
  if(state->allocatedDevices == 0) {
    THError("No OpenCL-enabled devices available");
  }
  if(state->currentDevice >= state->allocatedDevices || state->currentDevice < 0) {
    THError("Please use setDevice to choose an available device first");
  }
  if( state->clByDevice[device] == 0 ) {
    EasyCL *cl = 0;
    if(state->allowNonGpus) {
      cl = EasyCL::createForIndexedDevice(device);
    } else {
      cl = EasyCL::createForIndexedGpu(device);
    }
    state->clByDevice[device] = cl;
    THClScratchSpace *scratch = new THClScratchSpace();
    scratch->data = new float[FLOATS_PER_SCRATCH_SPACE];
    scratch->wrapper = cl->wrap(FLOATS_PER_SCRATCH_SPACE, scratch->data);
    scratch->wrapper->createOnDevice();
    state->scratchSpaceByDevice[device] = scratch;
    state->deviceInfoByDevice[device] = (DeviceInfo *)new easycl::DeviceInfo();
    if(state->allowNonGpus) {
      *((easycl::DeviceInfo *)state->deviceInfoByDevice[device]) = easycl::DevicesInfo::getDeviceInfo( device );
    } else {
      *((easycl::DeviceInfo *)state->deviceInfoByDevice[device]) = easycl::DevicesInfo::getGpuInfo( device );
    }
  }
  return state->clByDevice[device];
}

THClScratchSpace* THClState_getDeviceScratchSpace(THClState* state, int device, int stream)
{
  if(state->initialized == 0) {
    initializeState(state);
  }
  if( stream != 0 ) {
    THError("%d is not a stream", stream);
  }
  return state->scratchSpaceByDevice[device];
}

size_t THClState_getCurrentDeviceScratchSpaceSize(THClState* state)
{
  if(state->initialized == 0) {
    initializeState(state);
  }
  int device = state->currentDevice;
  return THClState_getDeviceScratchSpaceSize(state, device);
}

size_t THClState_getDeviceScratchSpaceSize(THClState* state, int device)
{
  if(state->initialized == 0) {
    initializeState(state);
  }

  return GLOBAL_SCRATCH_SPACE_PER_SM_STREAM; // true currently since we only have
             // one stream per device, currently
}

