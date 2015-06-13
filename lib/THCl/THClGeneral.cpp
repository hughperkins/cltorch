#include "THClGeneral.h"
#include "TH.h"

#include <stdio.h>
#include "EasyCL.h"

//#include "THCTensorRandom.h"
//#include "THCBlas.h"
//#include "THCAllocator.h"

void THClInit(THClState* state)
{
  printf("*******************************************\n");
  printf("THClInit()\n");
  state->allocatedDevices = easycl::DevicesInfo::getNumDevices();
  state->clByDevice = new EasyCL *[state->allocatedDevices];
  for(int i = 0; i < state->allocatedDevices; i++) {
    state->clByDevice[i] = 0;
  }
  state->currentDevice = 0;
  //state->cl = EasyCL::createForFirstGpuOtherwiseCpu(); // obviously this should change...
}

void THClShutdown(THClState* state)
{
  for(int i = 0; i < state->allocatedDevices; i++) {
    delete state->clByDevice[i];
  }
  delete state->clByDevice;
  printf("THClShutdown()\n");
  printf("*******************************************\n");
}

std::ostream &operator<<( std::ostream &os, const dim3 &obj ) {
  os << "dim3{" << obj.vec[0] << ", " << obj.vec[1] << ", " << obj.vec[2] << "}";
  return os;
}

int THClState_getNumDevices(THClState* state) {
  return state->allocatedDevices;
}
void THClState_setDevice(THClState* state, int device) {
  state->currentDevice = device;
}
int THClState_getDevice(THClState* state) {
  return state->currentDevice;
}
EasyCL *THClState_getCl(THClState* state) {
  if( state->clByDevice[state->currentDevice] == 0 ) {
    state->clByDevice[state->currentDevice] = EasyCL::createForIndexedDevice(state->currentDevice);
  }
  return state->clByDevice[state->currentDevice];
}

